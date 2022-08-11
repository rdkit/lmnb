from functools import reduce


def _sum_dicts(dict_list):
    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + 1
        return accumulator

    return reduce(reducer, dict_list, {})
