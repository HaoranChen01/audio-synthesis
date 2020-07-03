try:
    1/0
except ZeroDivisionError:
    print('fuck')
finally:
    print('ok')
print('continue')