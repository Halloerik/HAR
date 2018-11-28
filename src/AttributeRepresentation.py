import numpy as np
from scipy.spatial import distance

class attributeRepresentation():
    def __init__(self, numberOfClasses,numberOfAttributes):
        self.numberOfAttributes = numberOfAttributes
        self.numberOfClasses = numberOfClasses
        self.attributes = np.zeros((numberOfClasses,numberOfAttributes))
        
    def mutate(self): #TODO: mutate attributerepresentation
        pass
    
    def closest_class(self,attributeVector, distancemetric="cosine"):
        closestClass = 0
        if distancemetric is "cosine":
            closestDistance = distance.cosine(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = distance.cosine(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
                
        elif distancemetric is "euclidean":
            closestDistance = distance.euclidean(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = distance.euclidean(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
            
        elif distancemetric is "cityblock":
            closestDistance = distance.cityblock(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = distance.cityblock(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
            
        elif distancemetric is "braycurtis":
            closestDistance = distance.braycurtis(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = distance.braycurtis(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
            
        else:
            pass
        
    def attributevector_of_class(classnumber):
        return self.attributes[classnumber, :]