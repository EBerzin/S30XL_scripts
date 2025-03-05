import vxi11
instr=vxi11.Instrument("192.168.10.7")

print(instr.ask("*IDN?"))

def check_voltage(instr, ch):
    return instr.ask(":MEAS:VOLT? CH" + str(ch))

def check_current(instr, ch):
    return instr.ask(":MEAS:CURR? CH" + str(ch))

def channel_on(instr, ch):
    instr.write(":OUTP CH" + str(ch) + ",ON")

def channel_off(instr, ch):
    instr.write(":OUTP CH" + str(ch) + ",OFF")



#channel_on(instr,1)'
#instr.write(":OUTP CH1,OFF")
print(check_voltage(instr,2))
print(check_current(instr,2))