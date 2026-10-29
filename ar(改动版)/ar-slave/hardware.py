import wiringpi
from wiringpi import GPIO


class LED():
    def __init__(self,
                 pin:int):
        self.pin=pin
        wiringpi.wiringPiSetup()
        wiringpi.pinMode(self.pin,GPIO.OUTPUT)

    def on(self):
        wiringpi.digitalWrite(self.pin,GPIO.HIGH)

    def off(self):
        wiringpi.digitalWrite(self.pin,GPIO.LOW)


class Motor():
    def __init__(self,
                 pin:int):
        self.pin=pin
        wiringpi.wiringPiSetup()
        wiringpi.pinMode(self.pin,1)

    def start(self):
        wiringpi.softPwmCreate(self.pin,0,100)
         
    def set_duty(self,duty):
         wiringpi.softPwmWrite(self.pin,duty)
    
    
if __name__=="__main__":
    led=LED(pin=5)
    pwm=Motor(pin=1)
    pwm.start()
    pwm.set_duty(30)
    while True:
        led.on()
        wiringpi.delay(10) # Delay for 0.1 seconds
        led.off()
        wiringpi.delay(10)
