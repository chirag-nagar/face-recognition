#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import VGG16


# In[2]:


# load the VGG16 model without the top or FC layers
bottom_model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (224,224,3))


# In[3]:


bottom_model.input


# In[4]:


# here we check layes status
for i in range(len(bottom_model.layers)):
    print(str(i) + " "+ bottom_model.layers[i].__class__.__name__, bottom_model.layers[i].trainable)


# In[5]:


# here we make layers freeze
for layer in bottom_model.layers:
    layer.trainable = False


# In[6]:


# now again check status of layers
for i in range(len(bottom_model.layers)):
    print(str(i) + " "+ bottom_model.layers[i].__class__.__name__, bottom_model.layers[i].trainable)


# In[7]:


bottom_model.output


# In[8]:


from keras.layers import Dense,Flatten


# In[9]:


# here we create new layers for top model
top_model = bottom_model.output
top_model = Flatten()(top_model)
top_model = Dense(units = 128, activation = "relu")(top_model)
top_model = Dense(units = 64, activation = "relu")(top_model)
top_model = Dense(units = 32, activation = "relu")(top_model)
top_model = Dense(units = 10, activation = "softmax")(top_model)


# In[10]:


#top_model.output


# In[11]:


from keras.models import Model


# In[12]:


# here we add new layers (top model) layers to bottom model and make a new model
new_model = Model(inputs=bottom_model.input, outputs=top_model)


# In[13]:


# check new model is properly created
for (i,layer) in enumerate(new_model.layers):
    print(str(i) +" "+ layer.__class__.__name__, layer.trainable)    


# In[14]:


new_model.summary()


# In[15]:


from keras.optimizers import Adam


# In[17]:


new_model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.1),
              metrics = ['accuracy'])


# In[ ]:


new_model.fit()

