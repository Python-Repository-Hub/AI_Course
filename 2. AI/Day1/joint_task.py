import util
import wordsegUtil

_X_ = None

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start_state(self):
        # position before which text is reconstructed & previous word
        return 0, wordsegUtil.SENTENCE_BEGIN

    def is_end(self, state):
        return state[0] == len(self.query)

    def succ_and_cost(self, state):
        prev_pos, prev_word = state
        for step_size in range(1, len(self.query) - prev_pos + 1):
            next_pos = prev_pos + step_size
            vowel_removed_word = self.query[prev_pos: next_pos]
            for fill in self.possibleFills(vowel_removed_word):
                yield _X_, _X_, _X_  # return action, state, cost


if __name__ == '__main__':
    unigramCost, bigramCost = wordsegUtil.makeLanguageModels('leo-will.txt')
    smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
    possibleFills = wordsegUtil.makeInverseRemovalDictionary('leo-will.txt', 'aeiou')
    problem = JointSegmentationInsertionProblem('mgnllthppl', smoothCost, possibleFills)
    # problem = JointSegmentationInsertionProblem(wordsegUtil.removeAll('whatsup', 'aeiou'), smoothCost, possibleFills)

    import dynamic_programming_search
    dps = dynamic_programming_search.DynamicProgrammingSearch(verbose=1)
    # dps = dynamic_programming_search.DynamicProgrammingSearch(memory_use=False, verbose=1)
    # print dps.solve(problem)

    import uniform_cost_search
    ucs = uniform_cost_search.UniformCostSearch(verbose=0)
    print(ucs.solve(problem))


# === Other Examples ===
# 
# QUERIES_BOTH = [
#     'stff',
#     'hllthr',
#     'thffcrndprncndrw',
#     'thstffffcrndprncndrwmntdthrhrssndrdn',
#     'whatsup',
#     'ipovercarrierpigeon',
#     'aeronauticalengineering',
#     'themanwiththegoldeneyeball',
#     'lightbulbsneedchange',
#     'internationalplease',
#     'comevisitnaples',
#     'somethingintheway',
#     'itselementarymydearwatson',
#     'itselementarymyqueen',
#     'themanandthewoman',
#     'nghlrdy',
#     'jointmodelingworks',
#     'jointmodelingworkssometimes',
#     'jointmodelingsometimesworks',
#     'rtfclntllgnc',
# ]
