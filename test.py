# 两个链表的共同节点

def judge_list(list1,list2):
    l1 = []
    l2 = []
    list1_temp =list1
    list2_temp =list2
    while list1_temp!=null:
        l1.append(list1_temp)
        list1_temp= list1_temp.next
    while list2_temp!=null:
        l2.append(list2_temp)
        list2_temp= list2_temp.next
    
    if len(l1)==0 or len(l2)==0:
        return null

    flag_num = 0
    while len(l1)>0  and len(l2)>0:
        temp1 = l1.pop()
        temp2 = l2.pop()
        if temp1!=temp2:
            break
        flag_num+=1

    if flag_num==0:
        return null 
    if flag_num==len(l1):
        return list1
    if flag_num==len(l2):
        return list2

    flag_num = len(list1)-flag_num+1
    list1_temp =list1
    for i in range(len(list1))[:flag_num]:
        list1_temp= list1_temp.next
    return list1_temp




def judge_list2(linked_list1,linked_list2):
    linked_list1_length =0
    linked_list1_length =0
    ll1_temp = linked_list1
    ll2_temp = linked_list2

    while ll1_temp !=null:
        linked_list1_length+=1
    while ll2_temp !=null:
        linked_list2_length+=1

    ll1_temp = linked_list1
    ll2_temp = linked_list2
    if linked_list1_length>linked_list2_length:
        for i in rnage(linked_list1_length - linked_list2_length):
            ll1_temp= ll1_temp.next
    if linked_list2_length>linked_list1_length:
        for i in rnage(linked_list2_length - linked_list1_length):
            ll2_temp= ll2_temp.next

    node1 = 
    for i in range(len(min(linked_list1_length,linked_list2_length))):
        node1 = ll1_temp



s(n) = s(n-1)+s(n-2)

1 

2


for step in range(len(n)):


2^(n-1)