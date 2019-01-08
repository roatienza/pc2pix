# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os
import bpy
from math import radians
import mathutils
import math


sys.path.insert(0, '.')
sys.path.insert(0, '/usr/lib/python3.5/site-packages')

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--x', type=float, default=2,
                    help='camera x coordinate')
parser.add_argument('--y', type=float, default=2,
                    help='camera y coordinate')
parser.add_argument('--z', type=float, default=2,
                    help='camera z coordinate')
parser.add_argument('--i', type=int, default=0,
                    help='camera render index')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

camera_pos = [args.x, args.y, args.z]

bpy.context.scene.use_nodes = True

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=args.obj)
for obj in bpy.context.scene.objects:
    if obj.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = obj

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.energy = 1.0
lamp.shadow_method = 'NOSHADOW'
# Possibly disable specular shading:
lamp.use_specular = False

# set this value to the location of the camera
# location = mathutils.Vector([ 0.36861467, -1.8963581, -0.51763809])
location = mathutils.Vector(camera_pos)
focus_point=mathutils.Vector((0.0, 0.0, 0.0))
looking_direction = location - focus_point
rot_quat = looking_direction.to_track_quat('Z', 'Y')
bpy.data.objects['Lamp'].rotation_euler = rot_quat.to_euler()


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

# pixel resolution
scene = bpy.context.scene
scene.render.resolution_x = 128 # 224
scene.render.resolution_y = 128 # 224
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.camera

# best estimate of focal length
bpy.data.cameras[cam.name].lens = 58.

# interchange x and y from camera view.txt
# cam.location = [-0.8011148 , 1.7993333,  0.34729636]
# interchange -x and y
# Ex. cam.location = [1.7993333, 0.8011148, 0.34729636]

# 2nd test [[-1.35942933  1.35942933  0.55127471]]
cam.location = camera_pos # [ 0.36861467, -1.8963581, -0.51763809]

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

filepath = 'blender_render_{}_{}.png'.format(args.i, scene.render.resolution_x)
filepath = os.path.join(args.output_folder, filepath)
scene.render.filepath = filepath
bpy.ops.render.render(write_still=True)  # render still
