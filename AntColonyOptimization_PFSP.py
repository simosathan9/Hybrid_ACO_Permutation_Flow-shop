import math
import random
import copy

processing_time = {}
M = 5 # M refers to the number of machines
N = 20 # N refers to the number of jobs 
random.seed(13)

processing_time[0] = [54, 83, 15, 71, 77, 36, 53, 38, 27, 87, 76, 91, 14, 29, 12, 77, 32, 87, 68, 94]
processing_time[1] = [79, 3, 11, 99, 56, 70, 99, 60, 5, 56, 3, 61, 73, 75, 47, 14, 21, 86 , 5, 77]
processing_time[2] = [16, 89, 49, 15, 89, 45, 60, 23, 57,64, 7, 1, 63, 41, 63, 47, 26, 75, 77, 40]
processing_time[3] = [66, 58, 31, 68, 78, 91, 13, 59, 49, 85, 85, 9, 39, 41, 56, 40, 54, 77, 51, 31]
processing_time[4] = [58, 56, 20, 85, 53, 35, 53, 41, 69, 13, 86, 72, 8, 49, 47, 87, 58, 18, 68, 28]

#Initialize jobs to non scheduled
def initialize_unscheduled_jobs():
    unscheduled_jobs = []
    for j in range(0, N):
        unscheduled_jobs.append(j)
    return unscheduled_jobs
    
#Initialize pheromones to 10^(-6)
def init_pheromone_trails():
    pheromone_trails = []
    for i in range(0, N):
        pheromone_trails.append([])

    for i in range(0, N):
        for j in range(0, N):
            pheromone_trails[i].append(1e-06)
    return pheromone_trails

#Calculate heuristic information
#The heuristic information represents an a priori information about the problem instance definition provided by a source different from the ants.
e_j = []
for j in range(0, N):
    if processing_time[0][j] < processing_time[M-1][j]:
        e_j.append(1)
    else:
        e_j.append(-1)

S_j = []
for j in range(0, N):
    minimum = float('inf')
    for i in range(0, M - 1):
        m = processing_time[i][j] + processing_time[i + 1][j]
        if m < minimum:
            minimum = m
    S_j.append(e_j[j] / minimum)

heuristic_information = []
for i in range(0, N):
    heuristic_information.append([])
    for j in range(0, N):
        heuristic_information[i].append(S_j[j] + 0.51)

#Create the transition rule
def construct_pheromone_times_heuristic_info_matrix(t, h):
    #Initialize alpha and beta parameters(alpha and beta are two positive parameters denoting the relative importance of the pheromone trail versus the heuristic information.)
    #alpha=beta=1 from preliminary experiments
    alpha = 1
    beta = 1
    product_matrix = []
    for i in range(0, N):
        product_matrix.append([])
        for j in range(0, N):
            product_matrix[i].append(math.pow(t[i][j], alpha) * math.pow(h[i][j], beta))
    return product_matrix

#Finds the unscheduled job with the biggest value from the product matrix
def find_best_arg_from_product_matrix(product_matrix, unscheduled_jobs):
    max_val = -1 * float('inf')
    max_j = -1
    for i in range(0, M):
        for j in range(0, N):
            if j in unscheduled_jobs:
                if product_matrix[i][j] >= max_val:
                    max_val = product_matrix[i][j]
                    max_j = j
    return max_j

#Construct an initial solution
def create_initial_solution(pheromone_trails):
    #Initialize q0 parameter (additional values are (0.8, 0.85, 0.9 and 0.95))
    q0 = 0.99
    ant = []
    unscheduled_jobs = initialize_unscheduled_jobs()
    random_first_job = random.randint(0, M - 1)
    ant.append(random_first_job)
    unscheduled_jobs.remove(random_first_job)
    pr_matrix = construct_pheromone_times_heuristic_info_matrix(pheromone_trails, heuristic_information)
    while len(unscheduled_jobs) > 0:
        q = random.random()
        #Exploitation
        if q <= q0:
            max_j_arg = find_best_arg_from_product_matrix(pr_matrix, unscheduled_jobs)
            ant.append(max_j_arg)
            unscheduled_jobs.remove(max_j_arg)
        #Exploration
        else:
            i = len(ant)
            sum = 0
            probs = []
            for j in unscheduled_jobs:
                sum += pr_matrix[i][j]
            #Assign to each unscheduled job a probability
            #The probabibility is derived as follows
            #prob_ij = (pheromone_trail[i][j]^a * heuristic_information[i][j]^b) / sum(pheromone_trail[i][j]^a * heuristic_information[i][j]^b) for every j in unscheduled jobs
            for j in unscheduled_jobs:
                probs.append(pr_matrix[i][j] / sum)
            #Scale probabilities to sum up to 1
            scaled_probs = []
            scaled_probs.append(probs[0])
            for i in range(1, len(probs)):
                scaled_probs.append(probs[i] + scaled_probs[i - 1])
            #Get a random number that belongs to [0, 1)
            #Find the space that this random number belongs to and derive the unscheduled job that is randomly selected
            rand_probability = random.random()
            for i in range(0, len(scaled_probs)):
                if rand_probability <= scaled_probs[i]:
                    random_job_selected = unscheduled_jobs[i]
                    break
            ant.append(random_job_selected)
            unscheduled_jobs.remove(random_job_selected)
    return ant

#Calculate the makespan of a given solution
def calculate_makespan(sol):
    C_m1 = []
    C_1j = []
    for a in range(0, M):
        cm1 = 0
        for m in range(0, a+1):
            cm1 += processing_time[m][sol[0]]
        C_m1.append(cm1)    
    for b in range(0, N):
        c1j = 0
        for j in range(0, b+1):
            c1j += processing_time[0][sol[j]]
        C_1j.append(c1j) 
    
    C_mj = [] 
    for m in range(0, M):
        C_mj.append([])
        C_mj[m].append(C_m1[m])
    for j in range(1, N):
        C_mj[0].append(C_1j[j])
    for m in range(1, M):
        for j in range(1, N):
            C_mj[m].append(max(C_mj[m-1][j], C_mj[m][j - 1]) + processing_time[m][sol[j]])
    return C_mj[M - 1][N - 1]
    
def swap_move(ant: list, makespan):
    Pr = 0.7  # Probability threshold
    best_makespan = makespan
    best_solution = ant
    for j in range(0, len(ant)):
        job1 = ant[j]
        if random.random() <= Pr:
            for i in range(0, len(ant)):
                solution = copy.deepcopy(ant)
                if i == j:
                    continue
                else:
                    job2 = ant[i]
                    solution[i], solution[j] = job1, job2
                    move_makespan = calculate_makespan(solution)
                    if move_makespan < best_makespan:
                        best_makespan = move_makespan
                        best_solution = copy.deepcopy(solution)
    return best_solution

def global_updating_of_pheromone_trails(pheromone_trails, solution, makespan, is_better):
    #Update of pheromone trails is proportional to the quality of the makespan
    #The worse the makespan the smaller the term Z1 / makespan
    #Hence smaller quantity of pheromone is allocated to this arc
    rho = 0.25 #from preliminary experiments
    Z1 = 2
    Z2 = 10
    if is_better:
        for i in range(0, len(solution) - 1):
            j = i + 1
            pheromone_trails[solution[i]][solution[j]] = (1 - rho) * pheromone_trails[i][j] + rho * (Z2 / makespan)
    else:
        for i in range(0, len(solution) - 1):
            j = i + 1
            pheromone_trails[solution[i]][solution[j]] = (1 - rho) * pheromone_trails[i][j] + rho * (Z1 / makespan)
    return pheromone_trails
            

def ant_colony_opt_schema(iterations):
    num_of_iterations = 0
    best_makespan = 10e06
    best_solution = []
    pheromone_trails = init_pheromone_trails()
    while num_of_iterations < iterations:
        initial_solution = create_initial_solution(pheromone_trails)
        initial_makespan = calculate_makespan(initial_solution)
        new_solution = swap_move(initial_solution, initial_makespan)
        new_makespan = calculate_makespan(new_solution)
        #Use a threshold to accept worse quality solutions. Intensify as the algorithm proceeds.
        a = 50 
        while new_makespan < best_makespan + a:
            new_solution = swap_move(new_solution, new_makespan)
            new_makespan = calculate_makespan(new_solution)
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_solution = copy.deepcopy(new_solution)
            a = a - 2
        if new_makespan == best_makespan:
            pheromone_trails = global_updating_of_pheromone_trails(pheromone_trails, new_solution, new_makespan, True)
        else:
            pheromone_trails = global_updating_of_pheromone_trails(pheromone_trails, new_solution, new_makespan, False)
        num_of_iterations += 1
    return best_solution
    
for k in range(0, 30):   
    solution = ant_colony_opt_schema(50)
    cost_of_sol = calculate_makespan(solution)
    print('Cost of sequence : ', cost_of_sol)
    if k == 0:
        best_solution = copy.deepcopy(solution)
        best_cost = cost_of_sol
    else:
        if cost_of_sol <= best_cost:
            best_cost = cost_of_sol
            best_solution = copy.deepcopy(solution)
print('Best Sequence: ', best_solution)
print('Best cost: ', best_cost)