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

def main():
    bayes = generateModel()
    options = getopt.getopt(sys.argv[1:], "g:j:m:")[0]

    if not options:
        print "Need arguments to calculate probabilities"
        sys.exit()

    option = options[0][0]
    args = options[0][1]

    args = parseArgs(args)
    print args

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

    engine = JunctionTreeEngine(bayes)

if __name__ == "__main__":
    main()