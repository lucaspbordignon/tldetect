# tldetect
A Convolutional Neural Network to detect traffic lights states (red, green and yellow). A personal project in development.

# Training dataset
The dataset used to train de Conv Net is distributted by [Autti](http://autti.co). It contains 15,000 frames of a car driving in Mountain View California and neighboring cities during daylight conditions. It contains over 65,000 labels across all the frames, collected from a Point Grey research cameras running at full resolution of 1920x1200 at 2hz.

#### Labels 

- Car 
- Truck 
- Pedestrian
- Street Lights 

#### CSV Format
- frame 
- xmin
- ymin
- xmax
- ymax
- occluded
- label
- attributes (Only appears on traffic lights)

<table>
<tr>
    <td>Size</td>
    <td>3.3 GB</td>
</tr>
<tr>
    <td>Annotator</td>
    <td><a href="http://autti.co/">Autti</td>
</tr>
</table>

### [Download](http://bit.ly/udacity-annotations-autti)
