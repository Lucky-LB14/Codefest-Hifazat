#include <TinyGPS++.h>
#include <DFRobot_SIM808.h>
#include <TimeLib.h>  
#include <SoftwareSerial.h>
char lat[12];
char lon[12];
char wspeed[12];

char MESSAGE[300];

static const int PowerPin =9;

# define PHONE_NUMBER "+919667447952"

#define time_offset   19800  // define a clock offset of 3600 seconds (1 hour) ==> UTC + 1
 
// variable definitions
char Time[]  = "TIME: 00:00:00";
char Date[]  = "DATE: 00-00-2000";
byte last_second, Second, Minute, Hour, Day, Month;
int Year;
static const int RXPin = 10, TXPin = 11;
static const uint32_t GPSBaud = 9600;

// The TinyGPS++ object
TinyGPSPlus gps;


// The serial connection to the GPS device
SoftwareSerial ss(RXPin, TXPin);
DFRobot_SIM808 sim808(&ss);//Connect RX,TX,PWR,

void setup()
{
  pinMode(PowerPin, OUTPUT);
  //to turn on the GSM module -software switch instead Power Key
  digitalWrite(PowerPin, HIGH);
  delay (1000);
  digitalWrite (PowerPin, LOW); 
 
  
  Serial.begin(115200);
  ss.begin(GPSBaud);
      //******** Initialize sim808 module *************
  while(!sim808.init())
  {
      Serial.print("Sim808 init error\r\n");
      delay(1000);
  } 
 

  
  if( sim808.attachGPS())
      Serial.println("Open the GPS power success");
  else 
      Serial.println("Open the GPS power failure");
      
  Serial.println("Hifazat, We care for your Vehicle Safety :)");
 delay(3000);



}

void loop()
{



  if(Serial.available() >0)
  {
   String s = Serial.readString();

   if (s == "ss")
          //******** Initialize sim808 module *************
    while(!sim808.init()) {
        delay(1000);
        Serial.print("Sim808 init error\r\n");
    }
    Serial.println("Sim808 init success");
    Serial.println("Start to send message ...");
      float la = gps.location.lat();
      float lo = gps.location.lng();
      float ws = gps.speed.kmph();
 
      
      dtostrf(la, 4, 6, lat); //put float value of la into char array of lat. 6 = number of digits before decimal sign. 2 = number of digits after the decimal sign.
      dtostrf(lo, 4, 6, lon); //put float value of lo into char array of lon
      dtostrf(ws, 6, 6, wspeed);  //put float value of ws into char array of wspeed
 //put float value of ws into char array of wspeed
    
    
     sprintf(MESSAGE, " Date: %s\n Time: %s\nDriver is Lethargic and Not paying attention on road\n Latitude : %s\nLongitude : %s\nSpeed : %s kph\nhttp://maps.google.com/maps?q=%s,%s\n",Date,Time, lat, lon,wspeed, lat, lon);


    sim808.sendSMS(PHONE_NUMBER,MESSAGE);

    Serial.println("Let's Go drowsy...");
  
   
    
  }
  

  if(gps.speed.kmph() > 60.0)
  {
       //******** Initialize sim808 module *************
    while(!sim808.init()) {
        delay(1000);
        Serial.print("Sim808 init error\r\n");
    }
    Serial.println("Sim808 init success");
    Serial.println("Start to send message ...");
      float la = gps.location.lat();
      float lo = gps.location.lng();
      float ws = gps.speed.kmph();
      
      dtostrf(la, 4, 6, lat); //put float value of la into char array of lat. 6 = number of digits before decimal sign. 2 = number of digits after the decimal sign.
      dtostrf(lo, 4, 6, lon); //put float value of lo into char array of lon
      dtostrf(ws, 6, 6, wspeed);  //put float value of ws into char array of wspeed
    
 
    
    sprintf(MESSAGE, " Date: %s\n Time: %s\nDriver is rashly driving \n Latitude : %s\nLongitude : %s\nSpeed : %s kph\nhttp://maps.google.com/maps?q=%s,%s\n",Date,Time, lat, lon,wspeed, lat, lon);

    sim808.sendSMS(PHONE_NUMBER,MESSAGE);
    
    Serial.println("Let's Go speed...");
  

  }

 Serial.print("Vehicle_Speed:");
 printFloat(gps.speed.kmph(), gps.speed.isValid(), 6, 2);
 Serial.print("Vehicle_Lattitude:");
 printFloat(gps.location.lat(), gps.location.isValid(), 11, 6);
 Serial.print("Vehicle_Longitude:");
 printFloat(gps.location.lng(), gps.location.isValid(), 12, 6);
 Serial.print("Vehicle_Location_age:");
 printInt(gps.location.age(), gps.location.isValid(), 5);
      if (gps.time.isValid())
      {
        Minute = gps.time.minute();
        Second = gps.time.second();
        Hour   = gps.time.hour();
      }
 
      // get date drom GPS module
      if (gps.date.isValid())
      {
        Day   = gps.date.day();
        Month = gps.date.month();
        Year  = gps.date.year();
      }
 
      if(last_second != gps.time.second())  // if time has changed
      {
        last_second = gps.time.second();
 
        // set current UTC time
        setTime(Hour, Minute, Second, Day, Month, Year);
        // add the offset to get local time
        adjustTime(time_offset);
 
        // update time array
        Time[12] = second() / 10 + '0';
        Time[13] = second() % 10 + '0';
        Time[9]  = minute() / 10 + '0';
        Time[10] = minute() % 10 + '0';
        Time[6]  = hour()   / 10 + '0';
        Time[7]  = hour()   % 10 + '0';
 
        // update date array
        Date[14] = (year()  / 10) % 10 + '0';
        Date[15] =  year()  % 10 + '0';
        Date[9]  =  month() / 10 + '0';
        Date[10] =  month() % 10 + '0';
        Date[6]  =  day()   / 10 + '0';
        Date[7]  =  day()   % 10 + '0';
 
        // print time & date
        Serial.print("Day Today: ");
        print_wday(weekday());   // print day of the week
        Serial.println("");

        Serial.println(Time);         // print time (HH:MM:SS)

        Serial.println(Date); 
      }
 
if (gps.location.age() > 500)
{
  Serial.println("Vehicle is Parked");
}

  
  smartDelay(1000);

  if (millis() > 5000 && gps.charsProcessed() < 10)
    Serial.println(F("No GPS data received: check wiring"));
}

// This custom version of delay() ensures that the gps object
// is being "fed".
static void smartDelay(unsigned long ms)
{
  unsigned long start = millis();
  do 
  {
    while (ss.available())
      gps.encode(ss.read());
  } while (millis() - start < ms);
}

static void printFloat(float val, bool valid, int len, int prec)
{
  if (!valid)
  {
    while (len-- > 1)
      Serial.print('*');
    Serial.println(' ');
  }
  else
  {
    Serial.print(val, prec);
    int vi = abs((int)val);
    int flen = prec + (val < 0.0 ? 2 : 1); // . and -
    flen += vi >= 1000 ? 4 : vi >= 100 ? 3 : vi >= 10 ? 2 : 1;
    for (int i=flen; i<len; ++i)
      Serial.println(' ');
  }
  smartDelay(0);
}

static void printInt(unsigned long val, bool valid, int len)
{
  char sz[32] = "*****************";
  if (valid)
    sprintf(sz, "%ld", val);
  sz[len] = 0;
  for (int i=strlen(sz); i<len; ++i)
    sz[i] = ' ';
  if (len > 0) 
    sz[len-1] = ' ';
  Serial.println(sz);
  smartDelay(0);
}

static void printDateTime(TinyGPSDate &d, TinyGPSTime &t)
{
  if (!d.isValid())
  {
    Serial.print(F("********** "));
  }
  else
  {
    char sz[32];
    sprintf(sz, "%02d/%02d/%02d ", d.month(), d.day(), d.year());
    Serial.print(sz);
  }
  
  if (!t.isValid())
  {
    Serial.print(F("******** "));
  }
  else
  {
    char sz[32];
    sprintf(sz, "%02d:%02d:%02d ", t.hour(), t.minute(), t.second());
    Serial.println(sz);
  }

  //printInt(d.age(), d.isValid(), 5);
  smartDelay(0);
}

static void printStr(const char *str, int len)
{
  int slen = strlen(str);
  for (int i=0; i<len; ++i)
    Serial.print(i<slen ? str[i] : ' ');
  smartDelay(0);
}
void print_wday(byte wday)
{

  switch(wday)
  {
    case 1:  Serial.print(" SUNDAY  ");   break;
    case 2:  Serial.print(" MONDAY  ");   break;
    case 3:  Serial.print(" TUESDAY ");   break;
    case 4:  Serial.print("WEDNESDAY");   break;
    case 5:  Serial.print("THURSDAY ");   break;
    case 6:  Serial.print(" FRIDAY  ");   break;
    default: Serial.print("SATURDAY ");
  }
 
}
