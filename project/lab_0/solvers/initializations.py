import random

def init_greedy(instance, constraints):
    return NotImplementedError


def init_random(instance):
    shuffled_customers = instance.customers[:]
    random.shuffle(shuffled_customers)
    return shuffled_customers