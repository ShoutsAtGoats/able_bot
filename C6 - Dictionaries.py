# Dictionaries
# Also called Associative Arrays or Hashmaps

animal = {'name': 'Bob',
          'age': 12,
          'type': 'Rhino',
          'foods': ['Grass', 'Candy', 'Meatballs', 'Paintthinner']
          }

# Getting values from dictionary
print(animal['type'])
print(animal.get('Drinks', 'The item could not be found'))

# Adding to a dictionary
animal['weight'] = '800kg'
animal.update({'name': 'Rudy', 'age': 7, 'type': 'Tiger'})

# Removing from dictionary
del animal['weight']

# Looping through dictionaries
print(len(animal))
print(animal.keys())
print(animal.values())
print(animal.items())

# Not a great method
for key in animal:
    print(key)

# The right method
for key, value in animal.items():
    print(key, '--',  value)

