import sys
import getopt
import numpy
from pbnt.Inference import *


def generateModel():
    """Generate the disease model provided in the writeup
    """
   
    numberOfNodes = 5

    # Name the nodes
    pollution = 0
    smoker = 1
    cancer = 2
    xray = 3
    dyspnoea = 4

    pNode = BayesNode(0, 2, name="pollution")
    sNode = BayesNode(1, 2, name="smoker")
    cNode = BayesNode(2, 2, name="cancer")
    xNode = BayesNode(3, 2, name="xray")
    dNode = BayesNode(4, 2, name="dyspnoea")

    # Pollution
    pNode.add_child(cNode)

    # Smoker
    sNode.add_child(cNode)

    # Cancer
    cNode.add_parent(pNode)
    cNode.add_parent(sNode)
    cNode.add_child(xNode)
    cNode.add_child(dNode)

    # X-ray
    xNode.add_parent(cNode)

    # Dyspnoea
    dNode.add_parent(cNode)

    # Define the array of nodes
    nodes = [pNode, sNode, cNode, xNode, dNode]


    # Create distributions
    
    # Pollution distribution
    pDistribution = DiscreteDistribution(pNode)
    index = pDistribution.generate_index([],[])
    pDistribution[index] = [0.1, 0.9]
    pNode.set_dist(pDistribution)

    # Smoker distribution
    sDistribution = DiscreteDistribution(sNode)
    index = sDistribution.generate_index([],[])
    sDistribution[index] = [0.7, 0.3]
    sNode.set_dist(sDistribution)

    # Cancer
    dist = zeros([pNode.size(), sNode.size(), cNode.size()], dtype=float32)
    dist[0,0,] = [0.98,0.02]
    dist[1,0,] = [0.999,0.001]
    dist[0,1,] = [0.95, 0.05]
    dist[1,1,] = [0.97, 0.03]
    cDistribution = ConditionalDiscreteDistribution(nodes=[pNode, sNode, cNode], table=dist)
    cNode.set_dist(cDistribution)

    # X-ray
    dist = zeros([cNode.size(), xNode.size()], dtype=float32)
    dist[0,] = [0.8,0.2]
    dist[1,] = [0.1,0.9]
    xDistribution = ConditionalDiscreteDistribution(nodes=[cNode, xNode], table=dist)
    xNode.set_dist(xDistribution)

    # Dyspnoea
    dist = zeros([cNode.size(), dNode.size()], dtype=float32)
    dist[0,] = [0.7, 0.3]
    dist[1,] = [0.35,0.65]
    dDistribution = ConditionalDiscreteDistribution(nodes=[cNode, dNode], table=dist)
    dNode.set_dist(dDistribution)


    # Create bayes net
    return BayesNet(nodes)

def parseArgs(args):
    false = []
    true = []
    to_return = []

    negate = False
    for arg in args:
        if arg.isupper():
            to_return.append(arg)
        if arg == '~':
            negate = True
        elif arg in 'pscdx':
            if negate:
                false.append(arg)
                negate = False
            else:
                true.append(arg)

    return (to_return, true, false)

def marginalDistribution(bayes, engine, to_return, true_or_low, false_or_high):

    for node in bayes.nodes:
        if node.id == 0:
            pollution = node
        if node.id == 1:
            smoker = node
        if node.id == 2:
            cancer = node
        if node.id == 3:
            xray = node
        if node.id == 4:
            dyspnoea = node

    for evidence in true_or_low:
        if evidence == "p":
            engine.evidence[pollution] = True
        if evidence == "s":
            engine.evidence[smoker] = True
        if evidence == "c":
            engine.evidence[cancer] = True
        if evidence == "d":
            engine.evidence[dyspnoea] = True
        if evidence == "x":
            engine.evidence[xray] = True

    for evidence in false_or_high:
        if evidence == "p":
            engine.evidence[pollution] = False
        if evidence == "s":
            engine.evidence[smoker] = False
        if evidence == "c":
            engine.evidence[cancer] = False
        if evidence == "d":
            engine.evidence[dyspnoea] = False
        if evidence == "x":
            engine.evidence[xray] = False   

    for returns in to_return:
        if returns == "P":
            Q = engine.marginal(pollution)[0]
        if returns == "S":
            Q = engine.marginal(smoker)[0]
        if returns == "C":
            Q = engine.marginal(cancer)[0]
        if returns == "D":
            Q = engine.marginal(dyspnoea)[0]
        if returns == "X":
            Q = engine.marginal(xray)[0]

        index = Q.generate_index([False], range(Q.nDims))
        print "t =", 1 - Q[index]
        print "f =", Q[index]
        return Q[index]

def conditionalProbability(bayes, engine, conditional, true_or_low, false_or_high):
    
    for node in bayes.nodes:
        if node.id == 0:
            pollution = node
        if node.id == 1:
            smoker = node
        if node.id == 2:
            cancer = node
        if node.id == 3:
            xray = node
        if node.id == 4:
            dyspnoea = node

    for evidence in true_or_low:
        if evidence == "p":
            engine.evidence[pollution] = True
        if evidence == "s":
            engine.evidence[smoker] = True
        if evidence == "c":
            engine.evidence[cancer] = True
        if evidence == "d":
            engine.evidence[dyspnoea] = True
        if evidence == "x":
            engine.evidence[xray] = True

    for evidence in false_or_high:
        if evidence == "p":
            engine.evidence[pollution] = False
        if evidence == "s":
            engine.evidence[smoker] = False
        if evidence == "c":
            engine.evidence[cancer] = False
        if evidence == "d":
            engine.evidence[dyspnoea] = False
        if evidence == "x":
            engine.evidence[xray] = False   

    if conditional[0] == '~':
        index = False
        condition = conditional[1]
    else:
        index = True
        condition = conditional[0]


    if condition is 'p':
        condition = pollution
    if condition is 's':
        condition = smoker
    if condition is 'c':
        condition = cancer
    if condition is 'd':
        condition = dyspnoea
    if condition is 'x':
        condition = xray

    print index
    print condition.name

    Q = engine.marginal(condition)[0]
    index = Q.generate_index([index], range(Q.nDims))
    print "Conditonal Probability is: ", Q[index]
    return Q[index]


def main():
    bayes = generateModel()
    options = getopt.getopt(sys.argv[1:], "g:j:m:")[0]

    if not options:
        print "Need arguments to calculate probabilities"
        sys.exit()

    option = options[0][0]
    args = options[0][1]
    engine = JunctionTreeEngine(bayes)

    if option == '-m':
        to_return, true_or_low, false_or_high = parseArgs(args)
        marginalDistribution(bayes, engine, to_return, true_or_low, false_or_high)
    if option == '-g':
        conditional, given = args.split('|')
        to_return, true_or_low, false_or_high = parseArgs(given)
        conditionalProbability(bayes, engine, conditional, true_or_low, false_or_high)


if __name__ == "__main__":
    main()