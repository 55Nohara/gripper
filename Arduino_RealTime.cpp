#include "Wire.h"
#include <Adafruit_ADS1X15.h>

//-------check here before starting---------------
const float PWM_FREQ = 40;
const float CONTROL_FREQ = 20;         //[Hz]
const float CONTROL_DURATION = 64;     //[sec]
const float CONTROL_REST_TIME = 0;     //[sec]
const float CALIBRATION_DURATION = 2;  //[sec]
//const float INITIAL_PWMDC = 70;  //[%] 事前にキャリブレして求めた、初期のPWMDC, pythonとの通信を実装するのが面倒&不安定動作怖いので、ここで決めてしまう。
//------------------------------------------------

const int prescaler = 256;  // set this to match whatever prescaler value you set in CS registers below
const int maxInputColumn = 1250;
const unsigned long FRAME_RATE = (1 / CONTROL_FREQ) * 1000 * 1000;  // 1/[Hz] *1000: to miliSec further *1000 : to microsec
const byte numChars = 32;
uint8_t initial = 1;                   //1:initial procedure hasn't been done 0:procedure is done, do loop
uint8_t dataReady = 0;                 // 0:data is not ready(python is booting or received all data), 1:is ready
uint8_t calibration = 1;               // 1: do calibration, 0: calibration is done.
uint8_t adjActive = 1;                 // 0: frameTimeAdjust invalid 1: valid, この変数が0にされたループ(checkDataReadyが有効に働いた時)frameTimeAdjustを実質無効化するための変数
float input[maxInputColumn] = {};  //FREQ * DURATION
int currentInputIndex = 0;
int currentRealIndex = 0;
char receivedChars[numChars];  // an array to store the received data
boolean newData = false;

//strain sensor
int16_t readFromAds, read_StrainData_1, read_StrainData_2, read_object;
float volts, volts_strain_1, volts_strain_2, volts_object;
float pressureData, strainData_1, strainData_2, objectData;
Adafruit_ADS1115 ads;


void setup() {
  Serial.begin(115200);
  ads.setGain(GAIN_TWOTHIRDS);
  ads.begin();

  //2つのバルブを用いる時はpin2でも50と同じことをする
  pinMode(2, OUTPUT);  //pinMode(5, OUTPUT); pinMode(6, OUTPUT); pinMode(7, OUTPUT); pinMode(8, OUTPUT);

  int eightOnes = 255;  // this is 11111111 in binary
  // this operation (AND plus NOT), set the eight bits in TCCR registers to 0
  TCCR3A &= ~eightOnes;
  TCCR3B &= ~eightOnes;
  TCCR4A &= ~eightOnes;
  TCCR4B &= ~eightOnes;
  TCCR3A = _BV(COM3A1) | _BV(COM3B1);
  TCCR3B = _BV(WGM33) | _BV(CS32);
  TCCR4A = _BV(COM4A1) | _BV(COM4B1) | _BV(COM4C1);
  TCCR4B = _BV(WGM43) | _BV(CS42);
}

void loop() {
  unsigned long start_time = micros();

  if (initial == 1 && dataReady == 1) {
    // ここで２つのバルブを開ける→本来は制御が開始した瞬間に開けるべきなので、制御が開始するループの中で最も早いときに入れる。
    Serial.print('s');
    Serial.flush();
    sendSensorDataAndReceiveInput(currentInputIndex);
    generatePWM(PWM_FREQ, input[currentInputIndex]);
    initial = 0;
    currentInputIndex = 1;
    currentRealIndex = 1;
  } else if (dataReady == 1) {
    sendSensorDataAndReceiveInput(currentInputIndex);
    generatePWM(PWM_FREQ, input[currentInputIndex]);
    currentInputIndex++;
    currentRealIndex++;

    if (currentInputIndex == maxInputColumn - 1) currentInputIndex = 0;
    if (currentRealIndex == int(int(CONTROL_FREQ * (CONTROL_DURATION + CONTROL_REST_TIME)))) {  // つまり制御時間が終了したということ。
      clearValuables();
      clearInputBuffer();
      Serial.print('f');
      generatePWM(PWM_FREQ, 0);
      //ここで2バルブオフ
      //digitalWrite(49, LOW);
    }
  }

  checkDataReadyAndCalibrate();  //この関数で実際にDataReadyされた時はadjActive=0にする。
  frameTimeAdjust(start_time);
}  // end of loop()


//ここで実際にPWM波形を生成する
void generatePWM(float pwmfreq, float pwmDC1) {
  ICR3 = F_CPU / (prescaler * pwmfreq * 2);
  OCR3B = (ICR3) * (pwmDC1 * 0.01);
}

//未知の物体を把持する時に使う？
void sendSensorDataAndReceiveInput(int i) {  // i for index number
  Serial.print('t');
  Serial.flush();    // ここはシリアルバッファが満杯になったときのためにやっておいたほうがいいと思った。
  sendSensorData_3();  //send pressure and 2 strain 
  while (1) {
    recvWithEndMarker();                 //receive data as pwmDC1
    input[i] = showNewNumber(input[i]);  //receiveがすべて終わったときのみ-1以外を返す。
    if (input[i] != -1) {                //inputに入力が完全に終わったときのみ-1以外になる
      clearInputBuffer();                // just in case
      break;
    }
  }
}

void sendSensorData() {
  //int i = 0, averageNum = 5;
  // int16_t readFromAds, read_StrainData_1, read_StrainData_2, read_object;
  // float volts, volts_strain_1, volts_strain_2, volts_object;
  // float pressureData, strainData_1, strainData_2, objectData;

  //for (i=0; i<averageNum; i++){
  readFromAds = ads.readADC_SingleEnded(1);  //channel 1
  volts = ads.computeVolts(readFromAds);
  read_StrainData_1 = ads.readADC_SingleEnded(2);  //channel 2
  volts_strain_1 = ads.computeVolts(read_StrainData_1);
  read_StrainData_2 = ads.readADC_SingleEnded(3);  //channel 3
  volts_strain_2 = ads.computeVolts(read_StrainData_2);
  read_object = ads.readADC_SingleEnded(0);
  volts_object = ads.computeVolts(read_object);  //channel 0
  //}
  pressureData = 258.5625 * (volts - 0.5); // = 6.895 * (volts - 0.5) * (150.0 / (0.8 * 5.0));
  strainData_1 = (50 / volts_strain_1) - 10;  
  strainData_2 = (50 / volts_strain_2) - 10;
  objectData =  103.425 * (volts - 0.5); // = 6.895 * (volts_object - 0.5) * (60.0 / (0.8 * 5.0));  
  Serial.print(pressureData);
  Serial.print(',');
  Serial.print(strainData_1);
  Serial.print(',');
  Serial.print(strainData_2);
  Serial.print(',');
  Serial.print(objectData);
  Serial.print('\n');
  Serial.flush();
}

void sendSensorData_3() {
  //int i = 0, averageNum = 5;
  // int16_t readFromAds, read_StrainData_1, read_StrainData_2, read_object;
  // float volts, volts_strain_1, volts_strain_2, volts_object;
  // float pressureData, strainData_1, strainData_2, objectData;

  //for (i=0; i<averageNum; i++){
  readFromAds = ads.readADC_SingleEnded(1);  //channel 1
  volts = ads.computeVolts(readFromAds);
  read_StrainData_1 = ads.readADC_SingleEnded(2);  //channel 2
  volts_strain_1 = ads.computeVolts(read_StrainData_1);
  read_StrainData_2 = ads.readADC_SingleEnded(3);  //channel 3
  volts_strain_2 = ads.computeVolts(read_StrainData_2);
  //}
  pressureData = 258.5625 * (volts - 0.5); // = 6.895 * (volts - 0.5) * (150.0 / (0.8 * 5.0));
  strainData_1 = (50 / volts_strain_1) - 10;  
  strainData_2 = (50 / volts_strain_2) - 10;
  Serial.print(pressureData);
  Serial.print(',');
  Serial.print(strainData_1);
  Serial.print(',');
  Serial.print(strainData_2);
  Serial.print('\n');
  Serial.flush();
}

void checkDataReadyAndCalibrate() {
  if (dataReady == 0 && Serial.available() > 0) {
    char commandChar = Serial.read();
    Serial.print(commandChar);
    if (commandChar == 'e') {
      dataReady = 1;
      Serial.print(commandChar);
      //digitalWrite(50, LOW);  // 待ち時間前に減圧する。LOWにするので条件分岐は不要
      //digitalWrite(2, LOW);
      delay(3000);  // これ入れておかないと、なぜか無駄時間がステップで発生する。1000msくらい？100msだと28sampleくらいからまともに動き始める（これ以前の記録は、csvは常に上書きしているので、実質破棄している状態） 下の方においても、同じ様に動作する。
      calibrate();
      adjActive = 0;  //frameTimeAdjust無効化
    }
    clearInputBuffer();
  }
}

void calibrate() {
  int i = 0;
  clearInputBuffer();        // ここでやっても意味はないだろうけど、念の為。
  generatePWM(PWM_FREQ, 0);  //初期PWMDCを設定
  Serial.print('c');
  Serial.flush();  // キャリブレーション開始通知
  //c送る、ここでs+センサデータだけ送る、frameTimeAdjする、それを5秒間繰り返す、終了コマンド送る、pythonがオフセット計算する、pythonから計算終了指示来る→break, 通常制御開始
  for (i = 0; i < int(CONTROL_FREQ * CALIBRATION_DURATION); i++) {
    unsigned long start_time_calibration = micros();
    Serial.print('i');
    Serial.flush();
    sendSensorData();
    frameTimeAdjust(start_time_calibration);
  }
  Serial.print('m');
  Serial.flush();
  while (1) {
    if (Serial.available() > 0 && Serial.read() == 'm') {
      break;
    }
  }
  // おそらくは起きないだろうけど、もしお互いの同期的な問題が起きたらここ二delay(3000);とかを入れれば解決しそうな気がする。なお、ちゃんと考えたが、ここにdelayを入れるのは大丈夫。
}

//use this when get pwmDC1
void recvWithEndMarker() {
  static byte ndx = 0;  //word parmutaion
  char endMarker = '\n';
  char rc;

  if (Serial.available() > 0) {
    rc = Serial.read();  //get from serial
    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars) {
        ndx = numChars - 1;
      }
    } else {
      receivedChars[ndx] = '\0';  // terminate the string
      ndx = 0;
      newData = true;
    }
  }
}

float showNewNumber(float input) {
  if (newData == true) {          // new for this version
    input = atof(receivedChars);  // new for this version
    newData = false;
    return input;
  }
  return -1;
}

void frameTimeAdjust(unsigned long start_time) {
  boolean isTimeout = true;
  if (adjActive == 0) {  //adjActiveが0にされたループではstart_timeがこの時点での値になるので、タイムアウト判定はまあ開けられるだろう。
    start_time = micros();
    adjActive = 1;
  }
  if (micros() - start_time >= FRAME_RATE) {
    Serial.print('q');
    Serial.flush();
    return;
  }
  while (isTimeout) {
    if (micros() - start_time >= FRAME_RATE) {
      isTimeout = false;
      break;
    }
  }
}

void clearValuables() {
  initial = 1;
  dataReady = 0;
  calibration = 1;
  adjActive = 1;
  currentInputIndex = 0;
  currentRealIndex = 0;
  newData = false;
  for (int i = 0; i < maxInputColumn; i++) {
    input[i] = 0;
  }
}

void clearInputBuffer() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}
