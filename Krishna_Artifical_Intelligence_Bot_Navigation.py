#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def backtrack(t_start,t_end,t_mat,t_visit):
    visible = [(-1,0),(0,1),(1,0),(0,-1)]
    binary_mat = np.zeros((len(t_mat), len(t_mat)))
    binary_mat[t_end[0]][t_end[1]] = 1
    current = t_start
    traverse = []
    temp_traverse = []
    visited_stack = []
    visited_stack.append(t_start)
    for i in range(binary_mat.shape[0]):
         for j in range(binary_mat.shape[0]):
             cur = (i, j)
             if cur in t_visit:
                 binary_mat[i][j] = 1
    
    while current != t_end:
        for i in range(len(visible)):
            x = visible[i][0]+ current[0]
            y = visible[i][1]+ current[1]
            if x>-1 and y>-1 and x < binary_mat.shape[0] and y < binary_mat.shape[0]:
                if binary_mat[x][y] == 1 and (x,y) not in visited_stack:
                    temp_traverse.append((x, y))
        if not temp_traverse:
            ind = visited_stack.index(current) - 1
            current = visited_stack[ind]
            traverse.append(current)
        else:
            distance = []
            for i in range(len(temp_traverse)):   
                man = np.sqrt( ( (t_end[0] -temp_traverse[i][0]) ** 2) + ( (t_end[1] -temp_traverse[i][1]) ** 2) )
                distance.append(man)
            sort_dist = np.argsort(distance)
            current = temp_traverse[sort_dist[0]]
            traverse.append(current)
            visited_stack.append(current)
       
        temp_traverse = []
    traverse.pop()
    return(traverse)


def makemap(mat):
    count=0
    matrix=[]
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            #make 2D array of p,m,s as 1D
            mat=np.array(mat)
            matrix=mat.ravel()
            count+=1
    #initialize a list of elements with each element as a tuple of 2 values to 0
    maplist=list([[0 for x in range(2)] for y in range(count)])
    #map coordinates to list elements
    index=0
    for i in range(0,len(mat)):
        for j in range(0,len(mat)):
            maplist[index]=i,j
            index+=1
    return maplist,matrix   
      
def makedict(clist,mat,matrix,start):
     dictionary={}
     #use clist tuples as keys
     for i in range(len(clist)):
         dictionary[clist[i]]=0
     #get corresponding  p,m,s and their respective coordinates
     for i in range(len(matrix)):
         dictionary[clist[i]]=matrix[i]
     for i in dictionary.keys():
         if dictionary[i]=='p':
             dictionary[i]=10
         elif dictionary[i]=='m':
             dictionary[i]=100
         elif dictionary[i]=='s' :
             dictionary[i]=30
         else:
             dictionary[i]=-1
        #check if it's start and set cost as 0 
         if i==start:
             dictionary[i]=0  
     return dictionary
 
def get_direction(pre_node,current_node):
    x = pre_node[0] - current_node[0]
    y = pre_node[1] - current_node[1]
    direction = ''
    if x == 1 and y == 0:
        direction = 'N'
    elif x == -1 and y == 0:
        direction = 'S'
    elif x == 0 and y == 1:
        direction = 'W'
    elif x == 0 and y == -1:
        direction = 'E'
    return direction
    
#beam search implementation           
def solve(start,goal,mat):
    #goal test
    global visited
    direction = ""
    #call a function to map 2D list input to coordinate list
    clist,matrix=makemap(mat)
    #make a dictionary of coordinates and their costs of entering
    costdict=makedict(clist,mat,matrix,start)
    path.append(start)
    current=start
    #for i in range (0,15):
    while current != goal:
        current=find_frontier(current,costdict,clist,mat,goal,start)
    visited.append(current)
    
    for i in range(len(visited)-1):
        char = get_direction(visited[i],visited[i+1])
        direction = direction + char
    
    return direction 

    
def find_frontier(current,costdict,clist,mat,goal,start):
    #t_frontier changes at every step
    t_frontier={}
    global visited;
    global frontier;
    
    #t_visible nodes i.e 8
    t_visible={}
    front_cordinates = [(-1,0),(0,1),(1,0),(0,-1)]
    #set start as current node and add it to pathset
    for i in clist:
        x,y=current
        if x<(len(mat)-1):
            a=x+1,y
            t_frontier[(tuple(a))]=0  
            t_visible[(tuple(a))]=0  
        if x>0:
            a=x-1,y
            t_frontier[(tuple(a))]=0 
            t_visible[(tuple(a))]=0
        if y<(len(mat)-1):
            a=x,y+1
            t_frontier[(tuple(a))]=0  
            t_visible[(tuple(a))]=0
        if y>0:
            a=x,y-1
            t_frontier[(tuple(a))]=0 
            t_visible[(tuple(a))]=0
    
        #t_visible: adding diagonal elements
        if x<(len(mat)-1) and y<(len(mat)-1) :
            a=x+1,y+1
            t_visible[(tuple(a))]=0  
        if x>0 and y<(len(mat)-1) :
            a=x-1,y+1
            t_visible[(tuple(a))]=0
        if x>0 and y>0:
            a=x-1,y-1
            t_visible[(tuple(a))]=0
        if y>0 and x<(len(mat)-1):
            a=x+1,y-1
            t_visible[(tuple(a))]=0

    for i in costdict.keys():
        if i in t_visible.keys():
            t_visible[i]=costdict[i]

    for i in range(len(front_cordinates)):
        cor_a = (front_cordinates[i][0] + current[0], front_cordinates[i][1] + current[1]);
        cor_b = (front_cordinates[i-1][0] + current[0], front_cordinates[i-1][1] + current[1]);
        if cor_a in t_visible.keys() and cor_b in t_visible.keys():
            if t_visible[cor_a] == t_visible[cor_b] == -1:
                temp = (front_cordinates[i][0] + front_cordinates[i-1][0],front_cordinates[i][1] + front_cordinates[i-1][1])
                invisible_node = (current[0] + temp[0],current[1] + temp[1])
                del t_visible[invisible_node]
            
    for i in costdict.keys():
        if i in t_frontier.keys():
            t_frontier[i]=costdict[i]
            if t_frontier[i]==-1:
                t_frontier.pop(i, None)  
            
    #heuristic function
    visited.append(current)
    for key in list(t_frontier.keys()):
        if key in visited:
            del t_frontier[key]
            
    if goal in t_visible:
        goal_node = goal_heuristic(current,t_frontier,goal) 
        return goal_node
    elif not t_frontier:
        new_node = heuristic(frontier,goal)
        del frontier[new_node]
        bt_path = backtrack(current,new_node,mat,visited)
        for i in bt_path:
            visited.append(i)     
        return new_node
    else:
        cur=heuristic(t_frontier,goal)
        del t_frontier[cur]
        
        for key,value in list(t_frontier.items()):
            if key not in frontier:
                frontier[key] = value;
        
        t_frontier.clear()
        return cur
    
def goal_heuristic(current,t_frontier,goal):
    node_heu = []
    nodes = []
    if goal in t_frontier:
        return goal
    else:
        for k,v in t_frontier.items():
            nodes.append(k)
            cost = ( abs(k[0]-goal[0]) + abs(k[1]-goal[1]) ) * v
            node_heu.append(cost)
        sort_dist = np.argsort(node_heu)
        cur = nodes[sort_dist[0]]
        return cur    
            
    
def heuristic(t_frontier,goal):
    sort=[t_frontier[k] for k in sorted(t_frontier,key=t_frontier.__getitem__)]
    for k,v in t_frontier.items():
                if v==sort[0]:
                    path.append(k)
    l=len(path)
    cur=path[l-1]
    return cur


path=[]
visited = []
#dictionary for calculating next move
frontier = {}
visible = {}
  
def main():
    #take a size of square 2D matrix  and elements as m,p,s as input
    mat=[['w','w','w','w','w','w','w','p'],['m','m','s','s','s','p','p','w'],['s','s','p','s','p','s','m','w'],['p','w','m','s','s','s','w','w']
    ,['w','w','m','w','w','m','p','p'],['w','p','w','w','s','p','w','m'],['m','s','s','w','w','p','p','p'],['s','s','p','m','m','p','w','w']]
    #take input for start and goal state
    start=(6,7)
    goal=(6,2) 
    #call beam search function
    print(solve(start,goal,mat))
    
if __name__== "__main__":
  main()