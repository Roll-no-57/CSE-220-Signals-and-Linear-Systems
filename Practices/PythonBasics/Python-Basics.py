# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:51:47 2024

@author: Apurbo
"""
#Introduction
print("Hello")
print(type('10'))

#indentation
if True:
    print("true")
else:
    print("false")
    
print("hello")

#Qoutation
word = "hello"
sentence = "hello world"
paragraph = """    this is a 
               paragraph"""
num = 20.0
print(type(word))
print((sentence))
print((paragraph))
print(type(num))

#Comments
#Use # to single line comment
#use
"""
    triple quotes is used for multiple line commenting
"""

#User Input
# input("Press Enter key")
# print(input("Press Enter key"))
# print(type(input("Press Enter key")))
# n=int(input("Enter a number: "))
# print(type(n))
# print(n)

#Multiple statement on a single line
import sys ;x='foo';sys.stdout.write(x+'\n')

#print
print('string',end='') #doesn't print new line at the end
print('last word')

#multiple assignment
a=b=c=1
a,b,c = [1,2,'ohayo gozaimasu']
print(type(a),type(b),type(c))
a,b,c=1,3.0,"john"
print(a,b,c)
print(type(a),type(b),type(c))


#variable types
    #numbers
    #string
    #list
    #tuple
    #dictionary
    #no data type for characters .it is considered as a string of length 1
    #object type func: type(var)


#Numericals type

#integer type
a = 42      # Positive integer
b = -100    # Negative integer
c = 12345678901234567890  # Very large integer
print(type(a))  # Output: <class 'int'>

#floating type
x = 3.14   # A floating point number (real number)
y = -2.7   # Negative floating point number
z = 1.23e4 # Scientific notation (equivalent to 1.23 * 10^4)
print(type(x))  # Output: <class 'float'>

#Complex type
c1= 3+4j
c2= 2+6j
result = c1+c2
print(result)
print(type(result))

#Arithmatic of numericals
# Integers
result_int = 5 + 10  # 15
# Floats
result_float = 5.5 * 2.0  # 11.0
# Complex numbers
result_complex = (3 + 4j) * 2  # (6 + 8j)
print(result_complex)

#strings
    #strings are immutable in python
s='machine learning'
# s[7]=' ' this isn't allowed in python.

#Lists
"""Lists are C like arrays but can contain different objects in a single list"""
list = ["apurbo",786,2.23,'john',70.2]
list[2] = 'ami'
print(list)
ls = []
print(ls)

#Tuples
"""Tuples can't be modified"""

tuple=('abcd',786,2.23,'john',70.2)
print(type(tuple[0]))

#Operations on strings,list,tuple all are same
str = "helloworld"
print((str))
print(str[0])
print(str[2:5])
print(str[2:])
print(str[:5])
print(str[::2])
print(str[::-1])
print(str[::-2])
print(str*3)

#Python dictionary
dict ={}
dict['key1']='value1'
dict[2]="Yamete kudasai"
print(dict)
tynidict = {'name':'john',5:6734,'dept':'sales'}
print(tynidict)
print(tynidict.keys())
print(tynidict.values())

#DataType conversion

# Convert float to int
a = int(3.99)  # a = 3 (truncates the decimal part)
print(a)

# Convert string to int with base 10
b = int("42")  # b = 42
print(b)

# Convert binary string to int
c = int("1010", 2)  # c = 10 (binary '1010' is 10 in decimal)
print(c)


# Convert float to complex number
c = complex(3.0, 4.0)  # c = (3+4j)
print(c)

# Convert integer to complex
d = complex(5)  # d = (5+0j)
print(d)












