import math,itertools,time,heapq,threading,concurrent.futures,pulp
from itertools import combinations
from math import comb

def a1(n,s,k):
    m=pulp.LpProblem('A',pulp.LpMinimize)
    B=list(combinations(range(n),k))
    x=pulp.LpVariable.dicts('x',range(len(B)),0,1,pulp.LpBinary)
    m+=pulp.lpSum(x[i] for i in range(len(B)))
    for sub in combinations(range(n),s):
        idx=[i for i,b in enumerate(B) if set(sub)<=set(b)]
        m+=pulp.lpSum(x[i] for i in idx)>=1
    m.solve(pulp.PULP_CBC_CMD(msg=False,timeLimit=600))
    return None if pulp.LpStatus[m.status]!='Optimal' else int(pulp.value(m.objective)+0.5)

def b1(n,k,j,s,y):
    if j<s or y<1 or y>comb(j,s):return None
    B=list(combinations(range(n),k))
    subs=list(combinations(range(n),s))
    J=list(combinations(range(n),j))
    x=pulp.LpVariable.dicts('x',range(len(B)),0,1,pulp.LpBinary)
    m=pulp.LpProblem('B',pulp.LpMinimize)
    m+=pulp.lpSum(x[i] for i in range(len(B)))
    if y==1:
        for Jset in J:
            idx=[i for i,b in enumerate(B) if len(set(b)&set(Jset))>=s]
            m+=pulp.lpSum(x[i] for i in idx)>=1
    else:
        C=[pulp.LpVariable.dicts(f'c{t}',range(len(subs)),0,1,pulp.LpBinary) for t in range(len(J))]
        for t,Jset in enumerate(J):
            sids=[i for i,sub in enumerate(subs) if set(sub).issubset(set(Jset))]
            for sid in sids:
                idx=[i for i,b in enumerate(B) if set(subs[sid]).issubset(set(b))]
                m+=pulp.lpSum(x[i] for i in idx)>=C[t][sid]
            m+=pulp.lpSum(C[t][sid] for sid in sids)>=y
    m.solve(pulp.PULP_CBC_CMD(msg=False,timeLimit=600))
    return None if pulp.LpStatus[m.status]!='Optimal' else int(pulp.value(m.objective)+0.5)

def r1(n,k,j,s,y):
    return a1(n,s,k) if y=='all' else b1(n,k,j,s,int(y))

def cov(b,s):return set(combinations(b,s))

def a2(n,s,k):
    if s>k:return None
    B=[tuple(b) for b in combinations(range(n),k)]
    C=[cov(b,s) for b in B]
    subs=set(combinations(range(n),s))
    sol=[]
    while subs:
        idx=max(range(len(B)),key=lambda i:len(C[i]&subs))
        if not C[idx]&subs:break
        sol.append(idx)
        subs-=C[idx]
    return len(sol) if not subs else None

def b2(n,k,j,s,y):
    if j<s or y<1 or y>comb(j,s):return None
    B=[set(b) for b in combinations(range(n),k)]
    J=[set(t) for t in combinations(range(n),j)]
    sol=[]
    while True:
        uns=[Jset for Jset in J if sum(1 for b in sol if len(b&Jset)>=s)<y]
        if not uns:break
        idx=max(range(len(B)),key=lambda i:sum(1 for Jset in uns if len(B[i]&Jset)>=s))
        sol.append(B[idx])
        if len(sol)>n:return None
    return len(sol)

def r2(n,k,j,s,y):
    return a2(n,s,k) if y=='all' else b2(n,k,j,s,int(y))

def main():
    n,k,j,s=map(int,[input(),input(),input(),input()])
    y=input().strip()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1=ex.submit(r1,n,k,j,s,y)
        f2=ex.submit(r2,n,k,j,s,y)
        try:
            ans=f1.result(timeout=10)
            print(ans if ans is not None else '无可行解')
        except concurrent.futures.TimeoutError:
            ans=f2.result()
            print(ans if ans is not None else '无可行解')

if __name__=='__main__':
    main()
