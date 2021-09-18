import numpy as np


class meanShiftSeg:

    def __init__(self, image, radiusSize):
        self.image = np.array( image, copy = True )
        assert (self.image.shape[2] == 3)
        self.radiusSize = 2**radiusSize
        self.outimage = np.array( image, copy = True )
        self.colorSpace = np.zeros( (256,256) )
        self.nclusters = np.int(256/self.radiusSize)**2
        self.UVclust = np.zeros( shape=(self.nclusters, 2) )



    def __modifyColorSpace__(self):
     
        U = np.reshape( self.image[:,:,1], (-1,1) )
        V = np.reshape( self.image[:,:,2], (-1,1) )
        UV = np.transpose(np.array((U[:,0],V[:,0])))

        for u,v in UV :
                self.colorSpace[ u,v ] += 1


    def applyMeanShift(self):
        
        self.__modifyColorSpace__()
        rSize = self.radiusSize
        
        numOfWindPerDim = np.int(np.sqrt( self.nclusters ))
        clusters = []
        for itrRow in range( numOfWindPerDim ):
            for itrCol in range( numOfWindPerDim ):
                cntrRow, cntrCol = self.__windowIterator__(int(itrRow*rSize),int(itrCol*rSize)) 
               
                clusters.append( (cntrRow, cntrCol) )

        self.clustersUV = np.array( clusters )
        print (self.clustersUV)
        self.__classifyColors__()

        return self.outimage




    def __windowIterator__(self, row, col):
        
       
        rSize = self.radiusSize
        hrSize = rSize/2
        prevRow = 0
        prevCol = 0
       
        cluster = self.colorSpace[ row:row+rSize,col:col+rSize ]
        
        newRow, newCol = self.__center__( cluster )
        numOfIter = 0
        while( prevRow != newRow-hrSize and prevCol != newCol-hrSize ):
            if( numOfIter > np.sqrt(self.nclusters) ):
                break

            prevRow = newCol-hrSize
            prevCol = newCol-hrSize

            nxtRow = int((prevRow+row)%(256-rSize))
            nxtCol = int((prevCol+col)%(256-rSize))
            cluster = self.colorSpace[ nxtRow:nxtRow+rSize,nxtCol:nxtCol+rSize ]
            newRow, newCol = self.__center__( cluster )
            numOfIter += 1
        return row + newRow, col + newCol

    def __classifyColors__(self):
            
            rSize = self.radiusSize
            numOfWindPerDim = np.int(np.sqrt( self.nclusters ))
            for row in range( self.image.shape[0] ):
                for col in range( self.image.shape[1] ):
                    pixelU = self.outimage[row,col,1]
                    pixelV = self.outimage[row,col,2]
                    windowIdx = np.int( np.int(pixelV/rSize)  + np.int(numOfWindPerDim*( pixelU/rSize )))
                    self.outimage[row,col,1] = self.clustersUV[windowIdx, 0]
                    self.outimage[row,col,2] = self.clustersUV[windowIdx, 1]



    def __center__(self, cluster):

        momntIdx = range( self.radiusSize )
        totalMass = np.max(np.cumsum( cluster ))
        if (totalMass == 0):
            return self.radiusSize/2 , self.radiusSize/2
        if ( totalMass > 0 ):
            momentCol = np.max(np.cumsum(cluster.cumsum( axis=0 )[self.radiusSize-1]*momntIdx))
            cntrCol = np.round(1.0*momentCol/totalMass)
            momentRow = np.max(np.cumsum(cluster.cumsum( axis=1 )[:,self.radiusSize-1]*momntIdx))
            cntrRow = np.round(1.0*momentRow/totalMass)

            return cntrRow, cntrCol
    def __Eclid__(self, row, col):
        
        dist = np.sqrt( (row**2 + col**2 ))
        dist = np.round( dist )
        return dist

    def outimage(self):
        return self.outimage