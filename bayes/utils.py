from functools import reduce


def _sum_sets(set_list):
    def reducer(accumulator, element):
        for key in element:
            accumulator[key] = accumulator.get(key, 0) + 1
        return accumulator

    return reduce(reducer, set_list, {})
