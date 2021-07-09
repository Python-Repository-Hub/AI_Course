import util

ll = util.LinkedList.create_list(5, 4, 3, 2, 1)
print(ll)
print(ll.value)  # head's value

ll2 = ll.construct(6)  # constructtruct a new list with a value 6
print(ll2)

ll3 = ll.tail  # extract a sub-list that doesn't contain the head
print(ll3)

ll4 = ll3.construct(7)
print(ll4)

print(ll.tail is ll4.tail is ll2.tail.tail)  # linked lists share the sub-list
