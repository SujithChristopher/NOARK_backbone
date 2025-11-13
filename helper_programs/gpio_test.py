import gpiod

sync_pin = 17
chip = gpiod.Chip("gpiochip4")
sync_line = chip.get_line(sync_pin)
# sync_line.set_value(0)
sync_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)


while True:
    sync_state = sync_line.get_value()
    print(sync_state)

    # if sync_state:
    #     print('1', end='\r')
    #     print('aksjdfhkjsdh')

    # else:
    #     print('0', end='\r')
