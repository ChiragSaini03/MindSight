# MindSight

A Deep Learning and IOT project.


Making the user understand the environment approaches:-
Spatial Audio: Utilize spatial audio techniques to provide directional cues. Adjust the volume, pitch, or direction of the audio based on the location of detected objects relative to the user. This can help the user better understand the spatial layout of the environment.


Preciseness:-
8 dirs - monitoring object in these 8 dirs for feedback

        Top-Left      Top         Top-Right
        Left          Front       Right
        Bottom-Left   Bottom      Bottom-Right

The dir where the object's center lie will be marked as its dir and the feedback will be done accordingly.

Saving the current object's dir and depth and then if change in next frame will help in analysing the route of the person.

Avoiding a spam of feedback by saving the object depth and dir 
and only notfying if a new object has entered the frame or an existing object changed dir
or a significant amount of depth is changed.
