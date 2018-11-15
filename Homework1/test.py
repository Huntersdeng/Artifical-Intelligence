def walk(G,s,S=set()):
    P,Q=dict(),set()
    P[s]=None                    # s节点没有前任节点
    Q.add(s)                     # 从s开始搜索
    while Q:
        u=Q.pop()
        for v in G[u].difference(P,S):   # 得到新节点
            Q.add(v)
            P[v]=u               # 记录前任节点
    return P

def components(G):
    comp = []
    seen = set()
    for u in range(9):
        if u in seen: continue
        C = walk(G, u)
        seen.update(C)
        comp.append(C)
    return comp

if __name__ == "__main__":
    a, b, c, d, e, f, g, h, i= range(9)
    N = [
        {b, c, d},   # a
        {a, d},      # b
        {a,d},       # c
        {a,c,d},     # d
        {g,f},       # e
        {e,g},       # f
        {e,f},       # g
        {i},         # h
        {h}          # i
    ]
    comp = components(N)
    print(comp)
