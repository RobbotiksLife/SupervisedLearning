from fractions import Fraction


def find_multiplier(num):
    # Convert the float to a Fraction object to access numerator and denominator
    fraction = Fraction(num).limit_denominator()

    # Extract numerator and denominator
    numerator = fraction.numerator
    denominator = fraction.denominator

    # Calculate the least common multiple of numerator and denominator
    lcm = denominator

    # Find the least common multiple
    for i in range(1, max(numerator, denominator)):
        if lcm % numerator == 0:
            break
        lcm += denominator

    return lcm


def float_range(start: float, stop: float, step: float) -> [float]:
    multiplier = find_multiplier(step)
    return [x / multiplier for x in range(int(start*multiplier), int(stop*multiplier), int(step*multiplier))]

if __name__ == '__main__':
    print(float_range(-2, 2, 0.025))