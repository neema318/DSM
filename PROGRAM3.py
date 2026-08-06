#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
a = np.array ([[[2,4],[6,3]]])
b = np.array([[[4,8],[5,10]]])
print(a+b)
print(a-b)
print(a*b)
print (np.dot(a,b))
a @ b
print(np.transpose(a))


# In[21]:


import numpy as np
a = np.array([[3,1],[1,3]])
U,S,VT = np.linalg.svd(a)
print(a)
print(U)
print(S)
print(VT)
Sigma = np.diag(S)
a_reconstructed = np.dot(U,np.dot(Sigma,VT))
print(a_reconstructed)


# In[30]:


import matplotlib.pyplot as plt
x = [3,5,6,8,9]
y = [1,2,7,4,3]
plt.plot(x,y)
plt.title("Representation")
plt.xlabel("dist")
plt.ylabel("time")


# In[32]:


import matplotlib.pyplot as plt
subjects = ["English","DataScience","Mathematics"]
marks = [79,89,99]
plt.xlabel("subject")
plt.ylabel("marks")
plt.bar(subjects,marks)


# In[43]:


subjects = ["English","DataScience","Mathematics"]
marks = [79,56,99]
plt.xlabel("subject")
plt.ylabel("marks")
plt.scatter(subjects,marks)


# In[47]:



marks = [10,10,10,20,20,20,30,30,30,30,40,40,40,50,50,50]

plt.xlabel("marks")
plt.hist(marks)
plt.legend("profit")


# In[38]:


import matplotlib.pyplot as plt
subject = ["English","DataScience","Mathematics"]
marks = [79,56,99]
plt.pie(marks, labels=subject)
plt.show
plt.legend(marks)


# In[58]:


import matplotlib.pyplot as plt
x = [1,2,6,18]
y = [3,10,12,20]
plt.plot(x,y, 'r:o')
plt.show()


# In[60]:


import matplotlib.pyplot as plt
x = [1,2,6,18]
y = [3,10,12,20]
z = [1,3,5,7]
plt.subplot(1,2,1)
plt.plot(x,y)

x = [1,2,6,18]
y = [3,10,12,20]
z = [1,3,5,7]
plt.subplot(1,2,2)
plt.plot(x,y)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      


# In[36]:


import matplotlib.pyplot as plt
import numpy as np
men = [22,30,35,35,26]
women = [25,32,30,35,29]
x=np.arange(5)
plt.bar(x,men,0.3,label="men")
plt.bar(x+0.3,women,0.3,label="women")


# In[5]:


import matplotlib.pyplot as plt
x= [1,2,3,4]
y1=[2,4,6,8]
y2=[1,3,5,7]
plt.plot(x,y1,label="Line 1")
plt.plot(x,y2,label="Line 2")
plt.legend()


# In[14]:


import matplotlib.pyplot as plt
Language = ["Java","Python","PHP","JavaScript","C#","C++"]
Popularity = [22.2,17.6,8.8,8,7.7,6.7]
plt.scatter(Language,Popularity)


# In[26]:


import matplotlib.pyplot as plt
Language = ["Java","Python","PHP","JavaScript","C#","C++"]
Popularity = [22.2,17.6,8.8,8,7.7,6.7]
plt.bar(Language,Popularity)
plt.grid(True)


# In[27]:


import matplotlib.pyplot as plt
Language = ["Java","Python","PHP","JavaScript","C#","C++"]
Popularity = [22.2,17.6,8.8,8,7.7,6.7]
plt.pie(Popularity, labels=Language)
plt.show
plt.legend(Popularity,loc="upper right")


# In[ ]:





# In[ ]:




