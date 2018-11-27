class attributeRepresentation():
    def __init__(self, numberOfClasses,numberOfAttributes):
        self.numberOfAttributes = numberOfAttributes
        self.numberOfClasses = numberOfClasses
        self.attributes = numpy.zeros((numberOfClasses,numberOfAttributes))
        
    def mutate(): #TODO
        pass
    
    def closest_class(attributeVector, distancemetric="cosine"):
        closestClass = 0
        if distancemetric is "cosine":
            closestDistance = scipy.spatial.distance.cosine(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = scipy.spatial.distance.cosine(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
                
        elif distancemetric is "euclidean":
            closestDistance = scipy.spatial.distance.euclidean(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = scipy.spatial.distance.euclidean(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
            
        elif distancemetric is "cityblock":
            closestDistance = scipy.spatial.distance.cityblock(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = scipy.spatial.distance.cityblock(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
            
        elif distancemetric is "braycurtis":
            closestDistance = scipy.spatial.distance.braycurtis(attributeVector, self.attributes[0, :])
            for i in range(1,self.numberOfClasses):                
                classVector = self.attributes[i, :]
                newDistance = scipy.spatial.distance.braycurtis(attributeVector,classVector)
                if  newDistance < closestDistance:
                    closestClass = i
                    closestDistance = newDistance
            return closestClass
            
        else:
            pass
        
    def attributevector_of_class(classnumber):
        return self.attributes[classnumber, :]