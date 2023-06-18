
# Creating the list
lst = []
for idx in range(20):
    lst.append(idx)
# Slicing the list
print(lst)
print(lst[:5])
print(lst[5:])
print(lst[5:9])

# List Methods
print('Length: ', len(lst))
lst.append('Banana')
print(lst)
popped_item = lst.pop(5)
print(popped_item)
print(lst)
lst.remove('Banana')
print(lst)
print(sorted(lst, reverse=True))
print(max(lst))
print(min(lst))

# Nested Lists
new_lst = ['Bear', 'Tiger', 'Horse', 'Unicorn', 'Monkey']
lst.append(new_lst)
print(lst)
print(lst[19][0])

# Finding values
print(new_lst.index('Tiger'))
print('horse' in new_lst)

# Tuples
new_tuple = ('Bear', 'Tiger', 'Horse', 'Unicorn', 'Monkey')
print(type(new_tuple))
print(new_tuple)
print(new_tuple.count('Tiger'))

for item in new_tuple:
    print(item)