import numpy as np



class picture():
    def __init__(self, image, flattened_image, sol):
        self.image = image
        self.flattened_image = flattened_image
        self.sol = sol
        


def pictureGenerator(n, samplesize, center):
    solutions = np.random.randint(0,4, samplesize)
    solution_matrix = np.zeros((samplesize,4))
    S = np.zeros((n,n,samplesize))
    for sample in range(samplesize):
        solution_matrix[sample,solutions[sample]] = 1
        if solutions[sample] == 0:
            S[:,:,sample] = makeCircle(n, center)
        if solutions[sample] == 1:
            S[:,:,sample] = makeCross(n, center)
        if solutions[sample] == 2:
            S[:,:,sample] = makeSquare(n, center)
        if solutions[sample] == 3:
            S[:,:,sample] = makeLines(n, center)
        
    return S.transpose(), solution_matrix #want first index to images

def add_noice(S, noicerate):
    n = len(S[:,0])
    grid_width = len(S[0,:])
    for count, sample in enumerate(S):
        zero_frac = grid_width*grid_width
        ones_frac = int(grid_width*grid_width*noicerate)
        ranvec = np.array([1] * ones_frac + [0] * zero_frac)
        np.random.shuffle(ranvec)
        rangrid = np.zeros(np.shape(sample))
        for i in range(grid_width-1):
            rangrid[i,:] = ranvec[i*grid_width:(i+1)*grid_width]
        sample = (sample + rangrid)%2
        S[count] = sample
    return S

def ImageSetFlattener(S, n, samplesize):

    S_flattened = np.zeros((samplesize, n*n))
    for i in range(samplesize):
        S_flattened[i,:] = np.ravel(S[i,:])

    return S_flattened

def GenerateTestSet(n, samplesize, center = True, noicerate = 0.5):
    S, sol = pictureGenerator(n, samplesize, center)
    C = len(S)
    Snoice = add_noice(S, noicerate)
    S_flat_noice = ImageSetFlattener(Snoice, n, samplesize)
    picture_list = []
    for i in range(C):
        picture_list.append(picture(Snoice[i], S_flat_noice[i], sol[i]))
        
    return picture_list

    
def makeCircle(n, center):
    if center:
        r = (0,0)
    else:
        r = (np.random.randint(0,n//4),np.random.randint(0,n//4))
    grid = np.zeros((n,n))
    r0 = n/4
    r1 = n/3
    for i in range(n):
        for j in range(n):
            if r0 < np.linalg.norm(np.array([i - n//2,j -n//2])) < r1:
                grid[(i + r[0])%n][(j + r[1])%n] = 1
    return grid

def makeCross(n, center):
    if center:
        r = (0,0)
    else:
        r = (np.random.randint(0,n//3),np.random.randint(0,n//3))
    grid = np.zeros((n,n))
    r1 = n/3
    for i in range(n):
        for j in range(n):
            if abs(i-j + r[0] - r[1]) < 2:
                grid[i,j] = 1
            if abs(j + i - n + r[0] + r[1]) < 2:
                grid[i,j] = 1
    return grid

def makeSquare(n, center):
    if center:
        r = (0,0)
    else:
        r = (np.random.randint(0,n//3),np.random.randint(0,n//3))
    grid = np.zeros((n,n))
    centre = np.array([n//2,n//2])
    grid[r[0] + n//3:r[0] + 2*n//3,r[1] + n//3:r[1] + 2*n//3] = 1
    return grid

def makeLines(n, center):
    if center:
        r = (0,0)
    else:
        r = (np.random.randint(0,n),np.random.randint(0,n))
    grid = np.zeros((n,n))
    for i in range(n//4):
        grid[:, (i*n//4 + r[0])%n:(i*n//4 + r[0] + 1)%n] = 1
    return grid

def pictureGenerator_to_file(n, samplesize):
    solutions = np.random.randint(0,4, samplesize)
    solution_matrix = np.zeros((samplesize,4))
    S = np.zeros((n,n,samplesize))
    for sample in range(samplesize):
        solution_matrix[sample,solutions[sample]] = 1
        if solutions[sample] == 0:
            S[:,:,sample] = makeCircle(n)
        if solutions[sample] == 1:
            S[:,:,sample] = makeCross(n)
        if solutions[sample] == 2:
            S[:,:,sample] = makeSquare(n)
        if solutions[sample] == 3:
            S[:,:,sample] = makeLines(n)
        
    np.save("images.npy", S.transpose())#want first index to images