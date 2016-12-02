import numpy as np

def move(p,motion,p_move):
    #if(np.random.choice([0,1], p=[1-p_move, p_move])):
        #p = np.array(p)
        #p = np.roll(p, motion[0], axis=0)
        #p = np.roll(p, motion[1], axis=1)
        #p = p.tolist()
	for i in range(len(p)):
        	for j in range(len(p[0])):
            		p[i][j] = p_move*p[(i-motion[0]) % len(p)][(j-motion[1])%len(p[i])] + (1-p_move)*p[i][j]
    return p
   
def sense(p, colors, measurement, sensor_right):
    for i in range(len(colors)):
        for j in range(len(colors[0])):
            hit = (measurement == colors[i][j])
            p[i][j] = p[i][j] * (hit * sensor_right + (1-hit) * sensor_right)
    p = np.array(p)
    s = np.sum(p)
    p = p / s
    return p.tolist()
    

def localize(colors,measurements,motions,sensor_right,p_move):
    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]
    
    for i in range(len(colors[0])):
        p = move(p, motions[i], p_move)
        p = sense(p, colors, measurements[i], sensor_right)
    return p
    

def show(p):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x),r)) + ']' for r in p]
    print '[' + ',\n '.join(rows) + ']


colors = [['R','G','G','R','R'],
          ['R','R','G','R','R'],
          ['R','R','G','G','R'],
          ['R','R','R','R','R']]
measurements = ['G','G','G','G','G']
motions = [[0,0],[0,1],[1,0],[1,0],[0,1]]
p = localize(colors,measurements,motions,sensor_right = 0.7, p_move = 0.8)
show(p)
