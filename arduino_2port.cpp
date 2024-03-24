#include "Wire.h"
#include <Adafruit_ADS1X15.h>

//-------check here before starting---------------
const float PWM_FREQ = 40;
const float CONTROL_FREQ = 30;  //[Hz]
const float CONTROL_DURATION = 64;  //[sec]
const float CONTROL_REST_TIME = 0;  //[sec]
const float CALIBRATION_DURATION = 2;  //[sec]
const int VALVE_OPTION = 0;  // 0: no relief solenoid valve before control, 1: with relief solenoid 
//const float INITIAL_PWMDC = 70;  //[%] 事前にキャリブレして求めた、初期のPWMDC, pythonとの通信を実装するのが面倒&不安定動作怖いので、ここで決めてしまう。
//------------------------------------------------

const int prescaler = 256; // set this to match whatever prescaler value you set in CS registers below
const int maxInputColumn = 1250;
const unsigned long FRAME_RATE = (1 / CONTROL_FREQ) * 1000 * 1000; // 1/[Hz] *1000: to miliSec further *1000 : to microsec
const char fingerIndex = 't';
const byte numChars = 32;
int initial = 1;  //1:initial procedure hasn't been done 0:procedure is done, do loop
int dataReady = 0;  // 0:data is not ready(python is booting or received all data), 1:is ready
int calibration = 1;  // 1: do calibration, 0: calibration is done.
int adjActive = 1;  // 0: frameTimeAdjust invalid 1: valid, この変数が0にされたループ(checkDataReadyが有効に働いた時)frameTimeAdjustを実質無効化するための変数
float input[maxInputColumn] = {};  //FREQ * DURATION
int currentInputIndex = 0;
int currentRealIndex = 0;
char receivedChars[numChars];   // an array to store the received data
boolean newData = false;

const float CFACTOR = 0.18750;//空圧センサは0.5V~4.5Vの範囲で値を返す。50kPaまでしか絶対に帰ってこない設定なので、実際返すのは0.5V~2.5V→プラマイ4.096Vを設定すると、ビットあたりの読み取りは0.125mV, ビットあたりこれは0.00417kPaに相当する。→小数二桁までは分解能の面からは保証可能。
//strain sensor
Adafruit_ADS1115 ads;


void setup() {
  Serial.begin(115200);
  ads.setGain(GAIN_TWOTHIRDS);
  ads.begin();

//2つのバルブを用いる時はpin2でも50と同じことをする
  pinMode(2, OUTPUT); //pinMode(5, OUTPUT); pinMode(6, OUTPUT); pinMode(7, OUTPUT); pinMode(8, OUTPUT);
  pinMode(50, OUTPUT); //pinMode(50, OUTPUT);  // 疑似サーボ用/2ポート後の3pバルブ用。これらのピンには特にタイマーが割り当てられていない(=PWM出せない)っぽいので、このピンを採用した。
  
  int eightOnes = 255;  // this is 11111111 in binary
  // this operation (AND plus NOT), set the eight bits in TCCR registers to 0
  TCCR3A &= ~eightOnes; TCCR3B &= ~eightOnes; TCCR4A &= ~eightOnes; TCCR4B &= ~eightOnes;
  TCCR3A = _BV(COM3A1) | _BV(COM3B1); TCCR3B = _BV(WGM33) | _BV(CS32);
  TCCR4A = _BV(COM4A1) | _BV(COM4B1) | _BV(COM4C1); TCCR4B = _BV(WGM43) | _BV(CS42);

}

void loop() {
  unsigned long start_time = micros();
  
  

  if (initial == 1 && dataReady == 1) {
    // ここで２つのバルブを開ける→本来は制御が開始した瞬間に開けるべきなので、制御が開始するループの中で最も早いときに入れる。
    if (VALVE_OPTION==1){
      //digitalWrite(50, HIGH); // もし問題が起きたらポート直接操作に書き換える。ただ、digitalWriteって、5.5usくらいしか消費しないっぽいので、正直そこを一生懸命減らしても、バルブ応答時間(50000us)には大した影響は無いはず。
      //digitalWrite(2, HIGH); 
    }
    Serial.print('s'); Serial.flush();
    sendSensorDataAndReceiveInput(currentInputIndex);
    generatePWM(PWM_FREQ, input[currentInputIndex]);
    initial = 0; currentInputIndex = 1; currentRealIndex = 1;
  } else if (dataReady == 1) {
    sendSensorDataAndReceiveInput(currentInputIndex);
    generatePWM(PWM_FREQ, input[currentInputIndex]);
    currentInputIndex++; currentRealIndex++;
    
    if (currentInputIndex == maxInputColumn - 1)currentInputIndex = 0;
    if (currentRealIndex ==  int(int(CONTROL_FREQ * (CONTROL_DURATION+CONTROL_REST_TIME)))) {  // つまり制御時間が終了したということ。
      clearValuables(); clearInputBuffer();
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
  Serial.print(fingerIndex); Serial.flush(); // ここはシリアルバッファが満杯になったときのためにやっておいたほうがいいと思った。
  sendSensorData();  //send pressure and strain to PC
  while (1) {
    recvWithEndMarker(); //receive data as pwmDC1
    input[i] = showNewNumber(input[i]);  //receiveがすべて終わったときのみ-1以外を返す。
    if (input[i] != -1) {//inputに入力が完全に終わったときのみ-1以外になる
      clearInputBuffer();  // just in case
      break;
    }
  }
}


//input = pessureData, 
//add strainData here
void sendSensorData(){
  int i=0;
  int averageNum = 5;
  int16_t readFromAds;
  int16_t read_StrainData;
  float volts;
  float volts_strain;
  float pressureData;
  float strainData;

  //for (i=0; i<averageNum; i++){
  
  readFromAds = ads.readADC_SingleEnded(1);  //channel 0
  volts = ads.computeVolts(readFromAds);
  read_StrainData = ads.readADC_SingleEnded(2); //channel 1
  volts_strain = ads.computeVolts(read_StrainData);
  //}
  pressureData = 6.895*(volts - 0.5)*(150.0/(0.8*5.0)); 
  strainData = ((5-volts_strain)*10)/volts_strain;//計算式
  Serial.print(pressureData, 4);
  Serial.print(',');
  Serial.print(strainData, 4);
  Serial.print('\n'); Serial.flush();
  
}


void checkDataReadyAndCalibrate() {
  if (dataReady == 0 && Serial.available() > 0) {
    char commandChar = Serial.read();
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

void calibrate(){
  int i=0;
  clearInputBuffer();  // ここでやっても意味はないだろうけど、念の為。
  generatePWM(PWM_FREQ, 0);  //初期PWMDCを設定
  Serial.print('c'); Serial.flush();  // キャリブレーション開始通知
  //c送る、ここでs+センサデータだけ送る、frameTimeAdjする、それを5秒間繰り返す、終了コマンド送る、pythonがオフセット計算する、pythonから計算終了指示来る→break, 通常制御開始
  for (i=0; i<int(CONTROL_FREQ*CALIBRATION_DURATION); i++){
    unsigned long start_time_calibration = micros();
    Serial.print('i'); Serial.flush();
    
    sendSensorData();
    frameTimeAdjust(start_time_calibration);
  }
  Serial.print('m'); Serial.flush();
  while(1){
    if (Serial.available()>0 && Serial.read()=='m'){
      break;
    }
  }
  // おそらくは起きないだろうけど、もしお互いの同期的な問題が起きたらここ二delay(3000);とかを入れれば解決しそうな気がする。なお、ちゃんと考えたが、ここにdelayを入れるのは大丈夫。
}

//use this when get pwmDC1
void recvWithEndMarker() {
  static byte ndx = 0; //word parmutaion
  char endMarker = '\n';
  char rc;

  if (Serial.available() > 0) {
    rc = Serial.read(); //get from serial
    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars) {
        ndx = numChars - 1;
      }
    } else {
      receivedChars[ndx] = '\0'; // terminate the string
      ndx = 0;
      newData = true;
    }
  }
}

float showNewNumber(float input) {
  if (newData == true) {            // new for this version
    input = atof(receivedChars);   // new for this version
    newData = false;
    return input;
  }
  return -1;
}

void frameTimeAdjust(unsigned long start_time) {
  boolean isTimeout = true;
  if (adjActive == 0){  //adjActiveが0にされたループではstart_timeがこの時点での値になるので、タイムアウト判定はまあ開けられるだろう。
    start_time = micros();
    adjActive = 1;
  }
  if (micros() - start_time >= FRAME_RATE) {
    Serial.print('q'); Serial.flush();
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
  initial = 1; dataReady = 0; calibration = 1; adjActive = 1; currentInputIndex = 0; currentRealIndex = 0; newData = false;
  for (int i = 0; i < maxInputColumn; i++) {
      input[i] = 0;
  }
}

void clearInputBuffer() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}


