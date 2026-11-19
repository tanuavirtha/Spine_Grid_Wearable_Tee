# ESP32-S3 firmware

Add the Arduino sketch (`.ino`) used for the bench prototype here before publishing. It must stream one JSON line per sample at 115200 baud:

```json
{"upper":[ax,ay,az,gx,gy,gz],"left":[...],"right":[...],"flex1":2489,"flex2":1975}
```
