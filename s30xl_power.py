import vxi11
import time
import argparse


parser = argparse.ArgumentParser(
                    prog='s30xl_power.py',
                    description='Remotely controls the S30XL power supply.')


parser.add_argument('-ch', '--channel', type=int, default=None,
                    help = "Channel number")
parser.add_argument('-on', '--on', action='store_true', default=False,
                    help = "Turn channel on")
parser.add_argument('-off', '--off', action='store_true', default=False,
                    help = "Turn channel off")
parser.add_argument('-set_V', '--voltage', type=float, default=None,
                    help = "Set channel voltage")
parser.add_argument('-set_I', '--current', type=float, default=None,
                    help = "Set channel current")
parser.add_argument('-read', '--read', action='store_true', default=False,
                    help = "Read channel voltage and current")

def check_voltage(instr, ch):
    return instr.ask(":MEAS:VOLT? CH" + str(ch))

def check_current(instr, ch):
    return instr.ask(":MEAS:CURR? CH" + str(ch))

def set_voltage(instr, ch, voltage):
    instr.write(":SOUR" + str(ch) + ":VOLT " + str(voltage))

def check_set_voltage(instr, ch):
    return instr.ask(":SOUR" + str(ch) + ":VOLT?")

def check_set_current(instr, ch):
    return instr.ask(":SOUR" + str(ch) + ":CURR?")

def channel_on(instr, ch):
    instr.write(":OUTP CH" + str(ch) + ",ON")

def channel_off(instr, ch):
    instr.write(":OUTP CH" + str(ch) + ",OFF")

def main():
    args = parser.parse_args()

    instr=vxi11.Instrument("192.168.10.7")
    print(instr.ask("*IDN?"))
 
    if (args.channel is None):
        print("No channel number provided. Specify channel with -ch [#]")
        return
    if (args.read):
        print("Reading channel " + str(args.channel) + ":")
        print(check_voltage(instr,args.channel), "V")
        print(check_current(instr,args.channel), "A")
    if (args.voltage is not None):
        print("Channel " + str(args.channel) + " is currently reading: ")
        print(check_voltage(instr,args.channel), "V")
        print(check_current(instr,args.channel), "A")
        print("Channel " + str(args.channel) + " is " + str(instr.ask(":OUTP? CH%d"%(args.channel))))
        confirm = input("Confirm that you would like to set Channel " + str(args.channel) + " to " + str(args.voltage) + " V [y/n] : ")
        if (confirm == "y"):
            print("Setting voltage to " + str(args.voltage))
            set_voltage(instr, args.channel, args.voltage)
            print("Set points are:")
            print(check_set_voltage(instr,args.channel), "V")
            print(check_set_current(instr,args.channel), "A")
            time.sleep(1)
            print("Channel " + str(args.channel) + " is currently reading: ")
            print(check_voltage(instr,args.channel), "V")
            print(check_current(instr,args.channel), "A")
        else: return
    if (args.on) :
        print("Channel " + str(args.channel) + " is currently reading: ")
        print(check_voltage(instr,args.channel), "V")
        print(check_current(instr,args.channel), "A")
        print("Set points are:")
        print(check_set_voltage(instr,args.channel), "V")
        print(check_set_current(instr,args.channel), "A")
        confirm = input("Confirm that you would like to turn Channel " + str(args.channel) + " ON [y/n] : ")
        if (confirm == 'y'):
            channel_on(instr, args.channel)
            print("Channel " + str(args.channel) + " is ON.")
            time.sleep(1)
            print("Channel " + str(args.channel) + " is currently reading: ")
            print(check_voltage(instr,args.channel), "V")
            print(check_current(instr,args.channel), "A")
        else:
            return
    if (args.off):
        print("Channel " + str(args.channel) + " is currently reading: ")
        print(check_voltage(instr,args.channel), "V")
        print(check_current(instr,args.channel), "A")
        confirm = input("Confirm that you would like to turn Channel " + str(args.channel) + " OFF [y/n] : ")
        if (confirm == 'y'):
            channel_off(instr, args.channel)
            print("Channel " + str(args.channel) + " is OFF.")
            time.sleep(1)
            print("Channel " + str(args.channel) + " is currently reading: ")
            print(check_voltage(instr,args.channel), "V")
            print(check_current(instr,args.channel), "A")
        else:
            return


if __name__ == '__main__':
    main()