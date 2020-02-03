In the name of God

My name is Ahmad Rahimi and I was in the fourth semester of B.Sc of Computer Science in Sharif University of Technology when I built this repository.

In this repository i have uploaded some important and cool projects that we had to do as homework in Principles of Images Processing Taught by Dr Mostafa Kamali Tabrizi last semester (fall 2019).

Here I explain about each project a bit. For more information you can see the README file of each project and the codes.

- <b>Active Contour Segmentation</b>

    It is a segmentation algorithm for images in wich you have a list of points on the image forming a contour and you define an energy function for that contour (for the list of points) and the goal is to move the points to fit to a desired object in the image and for that we want to minimize the energy function of the contour. Both greedy and dynamic programming methods are implemented.


- <b>Book Cover Separation</b>
    
    In this project we are given an image of a number of books that have been photographed from above so the books are considered to be rectangles. The goal is to: 
    
    1 - Find the edge points
    
    2 - Find the coordinates of the 4 corners of the rectangle
    
    3 - Find a geometric transformation and warp the image to get the cover of each book
    
    Every three steps are implemented and the codes are available in different files. The second step has used a Hough Transform to find the points.
    

- <b>Cartoon Images</b>
    
    It is in fact a simple way of image segmentation in wich we run a SLIC algorithm to oversegment the image and then for each superpixel we define it's corresponding point as the average of intensity of the pixels in that superpixel at each channel. So for each superpixel we have a point in R<sup>3</sup> and then we perform a Mean-Shift algorithm to cluster these points. At the end the corresponding superpixels of each cluster form a segment of our image and we assign the values of the cluster center of that cluster as the intencities of the pixels in that segment. What we get is a sort of cartoon image and therefore I have called this project Cartoon Images.
    
 
- <b>Image Morphing</b>
    In this project we have two pictures and we want to make a movie to change the first picture to the second in a good way. In order to do that we first get some corresponding points in the images (e.g. if the images are the faces of two men, the tip of the nose, middle of both eyes, the corner of the lips, etc...). Then we perform a Delaunay algorithm to get the triangle mesh of the points. Then we move the points in the first image to their corresponding points in the second image slowly and for every image we want to make we warp the image in a way that the triangles in the triangle mesh created by Delaunay overlap and we get weighted average of two warped images. Two samples and codes are available in the folder.  
    

- <b>Texture Synthesizing and Texture Transfer</b>

    In Texture Synthesizing we want to build a bigger image out of a smaller image called sample without losing the resolution (we don't want to resize the image to a bigger one). For that purpose we first randomly select a patch and place it at the top left of the output pic. Then in every step we randomly select from the 10 patches of sample that matches better to the place of the output image we have already created. Then we find the min-cut path to cut the patch from and copy the remaining to the output image using Dynamic Programming. The results of making a bigger image out of an image of coffee beans and codes are available in the folder.
    In Texture Transfer we want to build an output image by puttin patches of a sample file together in a way that it is like a target image. For that purpose we use the same algorithm as above with the change that we add a similarity with target score to every patch and then randomly select from 10 high-scored patches. In the sample output I have made the picture of me and my friend by a Mat picture as sample picture.
