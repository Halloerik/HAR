import torch

import data_management


n_classes = 12
n_attributes = 12
distance_metric = "braycurtis"

attr_rep = data_management.Attribute_Representation(12,12)
attr_rep.diagonal_matrix()



for c in range(n_classes):
    class_tensor = torch.tensor([c])
    #print(c.shape)
    class_vector = attr_rep.attributevector_of_class(class_tensor)
    print("Class {} has vector {}".format(c, class_vector)) 
    
    predicted_class = attr_rep.closest_class(class_vector, distance_metric).item()
    print("Vector {} has predicted class {} with {}".format(class_vector,predicted_class,distance_metric))
    

print("\n")

labels = torch.tensor(range(n_classes))
print(labels)
class_vectors = attr_rep.attributevector_of_class(labels)
print(class_vectors)
predicted_classes = attr_rep.closest_class(class_vectors, distance_metric)
print(predicted_classes)
