#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import statistics
from functools import cmp_to_key  # kept in case you extend hull logic
sys.path.append('/usr/local/lib/python3.8/dist-packages')
import numpy as np
import Metashape


# --------------------------- Compatibility check ----------------------------
if not Metashape.app.version.startswith("1.8"):
    raise Exception(
        f"Incompatible Metashape version: {Metashape.app.version} (requires 1.8.x)"
    )


# --------------------------- Utilities --------------------------------------
def find_files(folder, exts):
    photos = []
    for root, _, files in os.walk(folder):
        for p in files:
            if os.path.splitext(p)[1].lower() in exts:
                photos.append(os.path.join(root, p))
    return photos


def create_chunk_from_folder(chunk_folder, doc):
    photos = find_files(chunk_folder, [".jpg", ".jpeg", ".tif", ".tiff"])
    if not photos:
        print(f"[WARN] No images found under: {chunk_folder}")
        return None
    chunk = doc.addChunk()
    chunk.addPhotos(photos)
    chunk.label = os.path.basename(os.path.normpath(chunk_folder))
    print(f"Chunk '{chunk.label}' created with {len(chunk.cameras)} images")
    doc.save()
    return chunk


def cross(a, b):
    # Returns normalized cross product as Metashape.Vector
    result = Metashape.Vector([a.y * b.z - a.z * b.y,
                               a.z * b.x - a.x * b.z,
                               a.x * b.y - a.y * b.x])
    return result.normalized()


# --------------------------- 2D Convex Hull + Min Area Rect -----------------
# Minimal helpers matching your original logic; consider SciPy if available.
link = lambda a, b: np.concatenate((a, b[1:]))
edge = lambda a, b: np.concatenate(([a], [b]))

def qhull2D(sample):
    """Very small convex-hull helper for 2D Nx2 array. Assumes non-degenerate input."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) <= 2:
        return sample

    axis = sample[:, 0]
    base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)

    # This is a simplified quickhull-like step; good enough if input already hull-ish.
    h, t = base
    dR = np.dot(sample - h, np.dot(((0, -1), (1, 0)), (t - h)))
    outer = sample[dR > 0]
    if len(outer) > 0:
        pivot = sample[np.argmax(dR)]
        left_points = sample[dR < 0]
        right_points = sample[dR > 0]
        base_left = edge(h, pivot)
        base_right = edge(pivot, t)
        left_result = link(qhull2D(left_points), qhull2D(base_left))
        right_result = link(qhull2D(right_points), qhull2D(base_right))
        return link(left_result, right_result)
    else:
        return base


def minBoundingRect(hull_points_2d):
    """Returns (angle, area, width, height, center_xy, corners_xy)"""
    hull_points_2d = np.asarray(hull_points_2d, dtype=float)
    if hull_points_2d.shape[0] < 2:
        raise ValueError("Not enough hull points to compute bounding rectangle.")

    edges = np.diff(hull_points_2d, axis=0)
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    edge_angles = np.abs(edge_angles % (math.pi / 2.0))
    edge_angles = np.unique(edge_angles)

    min_bbox = (0.0, sys.maxsize, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # angle, area, w, h, minx, maxx, miny, maxy

    for ang in edge_angles:
        R = np.array([[math.cos(ang), math.cos(ang - math.pi / 2.0)],
                      [math.cos(ang + math.pi / 2.0), math.cos(ang)]])
        rot = R @ hull_points_2d.T
        min_x, max_x = np.nanmin(rot[0]), np.nanmax(rot[0])
        min_y, max_y = np.nanmin(rot[1]), np.nanmax(rot[1])
        w, h = (max_x - min_x), (max_y - min_y)
        area = w * h
        if area < min_bbox[1]:
            min_bbox = (ang, area, w, h, min_x, max_x, min_y, max_y)

    # Re-create rotation matrix for min rect
    angle = min_bbox[0]
    R = np.array([[math.cos(angle), math.cos(angle - math.pi / 2.0)],
                  [math.cos(angle + math.pi / 2.0), math.cos(angle)]])
    min_x, max_x, min_y, max_y = min_bbox[4], min_bbox[5], min_bbox[6], min_bbox[7]

    center_xy_rot = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    center_xy = center_xy_rot @ R

    # corners (max_x,min_y) -> (min_x,min_y) -> (min_x,max_y) -> (max_x,max_y)
    corner_rot = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    corners_xy = corner_rot @ R

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_xy, corners_xy)


# --------------------------- Region Refinement ------------------------------
# def refine_region_from_cams(chunk):
#     """
#     Builds a minimal-area XY rectangle from enabled camera centers in the chunk CRS,
#     inflates by BUFFER% in XY and by ~10x Z spread, and sets chunk.region.
#     """
#     BUFFER = 15.0
#     new_region = Metashape.Region()

#     cams = [c for c in chunk.cameras if c.transform and c.enabled]
#     if not cams:
#         raise RuntimeError("No enabled cameras with transforms to refine region.")

#     T = chunk.transform.matrix
#     s = chunk.transform.scale()  # scalar
#     crs = chunk.crs

#     camxy = np.zeros((len(cams), 2), dtype=float)
#     camz = np.zeros(len(cams), dtype=float)

#     for i, cam in enumerate(cams):
#         c = crs.project(T.mulp(cam.center))  # CRS coords
#         camxy[i] = [c.x, c.y]
#         camz[i] = c.z

#     zmed = float(statistics.median(camz))
#     zmin = float(np.min(camz))
#     zmax = float(np.max(camz))

#     hull_points = qhull2D(camxy)[::-1]
#     rot_angle, area, width, height, center_xy, corners_xy = minBoundingRect(hull_points)

#     # Corners at zmax in CRS -> back to local coordinates
#     corners = [Metashape.Vector([x, y, zmax]) for (x, y) in corners_xy]
#     corners = [T.inv().mulp(crs.unproject(c)) for c in corners]

#     side1 = corners[0] - corners[1]
#     side2 = corners[0] - corners[-1]
#     side1g = T.mulp(corners[0]) - T.mulp(corners[1])
#     side2g = T.mulp(corners[0]) - T.mulp(corners[-1])
#     side3g = T.mulp(corners[0]) - T.mulp(Metashape.Vector([corners[0].x, corners[0].y, zmin]))

#     new_size = ((100.0 + BUFFER) / 100.0) * Metashape.Vector([
#         side2g.norm() / s,
#         side1g.norm() / s,
#         10.0 * side3g.norm() / s
#     ])

#     # Center in CRS at zmed -> back to local
#     center_point_local = T.inv().mulp(
#         crs.unproject(Metashape.Vector([center_xy[0], center_xy[1], zmed]))
#     )

#     # Rotation about Z by rot_angle as a Metashape.Matrix
#     c, si = float(np.cos(rot_angle)), float(np.sin(rot_angle))
#     Rz = Metashape.Matrix([[c, -si, 0],
#                            [si,  c, 0],
#                            [0,   0,  1]])

#     new_region.rot = Rz
#     new_region.center = center_point_local
#     new_region.size = new_size

#     chunk.region = new_region
#     return chunk.region

def _is_geographic(crs: Metashape.CoordinateSystem) -> bool:
    try:
        # Heuristic: geographic systems typically have angular XY units (degrees)
        u = crs.units()
        # units() returns a tuple like ('degree','degree','metre') or ('metre','metre','metre')
        return len(u) >= 2 and u[0].lower().startswith('degree') and u[1].lower().startswith('degree')
    except Exception:
        # Fallback: treat unknown as geographic to be safe
        return True

def _make_utm_crs_from_lonlat(lon_deg: float, lat_deg: float) -> Metashape.CoordinateSystem:
    # Compute UTM zone (EPSG 326xx for N, 327xx for S)
    zone = int((lon_deg + 180.0) // 6) + 1
    if lat_deg >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return Metashape.CoordinateSystem(f"EPSG::{epsg}")

def refine_region_from_cams(chunk):
    BUFFER = 15.0
    new_region = Metashape.Region()

    cams = [c for c in chunk.cameras if c.transform and c.enabled]
    if not cams:
        raise RuntimeError("No enabled cameras with transforms to refine region.")

    T   = chunk.transform.matrix
    s   = chunk.transform.scale()
    crs = chunk.crs

    # First, get lon/lat in WGS84 to decide on a projected temp CRS if needed
    wgs84 = Metashape.CoordinateSystem("EPSG::4326")
    lonlats = []
    for cam in cams:
        w = wgs84.project(T.mulp(cam.center))
        lonlats.append((w.x, w.y))

    mean_lon = float(np.mean([p[0] for p in lonlats]))
    mean_lat = float(np.mean([p[1] for p in lonlats]))

    # Choose the metric CRS for XY geometry
    if _is_geographic(crs):
        xy_crs = _make_utm_crs_from_lonlat(mean_lon, mean_lat)
    else:
        xy_crs = crs  # already projected

    # Build XY in metric CRS; keep Z as meters from original world
    camxy = np.zeros((len(cams), 2), dtype=float)
    camz  = np.zeros(len(cams), dtype=float)

    for i, cam in enumerate(cams):
        p_xy = xy_crs.project(T.mulp(cam.center))  # XY in meters if projected CRS
        camxy[i] = [p_xy.x, p_xy.y]
        # For Z, use original world Z (meters); if xy_crs is 2D, project() leaves z unchanged anyway.
        p_w = crs.project(T.mulp(cam.center))  # get z consistently from chunk CRS
        camz[i] = p_w.z

    zmed = float(statistics.median(camz))
    zmin = float(np.min(camz))
    zmax = float(np.max(camz))

    # Hull/rect in metric XY
    hull_points = qhull2D(camxy)[::-1]
    rot_angle, area, width, height, center_xy, corners_xy = minBoundingRect(hull_points)

    # Build corners at zmax in xy_crs, then unproject to chunk-local
    corners = [Metashape.Vector([x, y, zmax]) for (x, y) in corners_xy]
    # unproject from xy_crs back to world, then to local with T.inv()
    corners = [T.inv().mulp(xy_crs.unproject(c)) for c in corners]

    side1  = corners[0] - corners[1]
    side2  = corners[0] - corners[-1]
    side1g = T.mulp(corners[0]) - T.mulp(corners[1])
    side2g = T.mulp(corners[0]) - T.mulp(corners[-1])
    side3g = T.mulp(corners[0]) - T.mulp(Metashape.Vector([corners[0].x, corners[0].y, zmin]))

    new_size = ((100.0 + BUFFER) / 100.0) * Metashape.Vector([
        side2g.norm() / s,
        side1g.norm() / s,
        10.0 * side3g.norm() / s
    ])

    # Center from metric center_xy (xy_crs) at zmed (meters) back to local
    center_point_local = T.inv().mulp(
        xy_crs.unproject(Metashape.Vector([center_xy[0], center_xy[1], zmed]))
    )

    # Rotation about Z by angle measured in metric XY space
    c, si = float(np.cos(rot_angle)), float(np.sin(rot_angle))
    Rz = Metashape.Matrix([[c, -si, 0],
                           [si,  c, 0],
                           [0,   0,  1]])

    new_region.rot = Rz
    new_region.center = center_point_local
    new_region.size = new_size
    chunk.region = new_region
    return chunk.region


# --------------------------- Main ------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: general_workflow.py <image_folder> <project_folder>")
        raise SystemExit(2)

    image_folder = sys.argv[1]
    project_folder = sys.argv[2]

    os.makedirs(project_folder, exist_ok=True)
    output_folder = os.path.join(project_folder, 'output')
    os.makedirs(output_folder, exist_ok=True)

    log_path = os.path.join(output_folder, 'log.txt')
    logfile = open(log_path, 'a', buffering=1)  # line-buffered logging
    logfile.write(time.strftime("Process started at %Y-%m-%d %H:%M:%S\n"))

    doc = Metashape.Document()
    doc.save(os.path.join(project_folder, 'project.psx'))

    # Discover subdirectories and create a chunk per folder
    subdirs = [os.path.join(image_folder, d) for d in os.listdir(image_folder)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    for sd in subdirs:
        create_chunk_from_folder(sd, doc)

    # Process chunks
    for chunk in doc.chunks:
        print()
        print(f"Start processing chunk {chunk.label}")
        logfile.write(f"Started processing of {chunk.label}.\n")

        # Matching / alignment
        chunk.matchPhotos(
            downscale=0,
            keypoint_limit=50000,
            tiepoint_limit=25000,
            keypoint_limit_per_mpx=25000,
            generic_preselection=True,
            reference_preselection=True,  # set False if you don't have references
        )
        doc.save()

        chunk.alignCameras(adaptive_fitting=True)
        doc.save()

        # Region refinement
        try:
            chunk.region = refine_region_from_cams(chunk)
        except Exception as e:
            print(f"[WARN] Region refinement failed for {chunk.label}: {e}")

        # Tie point filtering
        reperr = 0.1
        recunc = 40

        f = Metashape.PointCloud.Filter()
        f.init(chunk, Metashape.PointCloud.Filter.ReprojectionError)
        f.removePoints(reperr)

        f = Metashape.PointCloud.Filter()
        f.init(chunk, Metashape.PointCloud.Filter.ReconstructionUncertainty)
        f.removePoints(recunc)

        try:
            tp_total = len(chunk.point_cloud.points)
            tp_valid = sum(1 for p in chunk.point_cloud.points if p.valid)
            tp_invalid = tp_total - tp_valid
            print(f"{tp_invalid} tie points filtered; {tp_valid} remaining")
        except Exception:
            pass

        chunk.optimizeCameras(fit_corrections=True, adaptive_fitting=True, tiepoint_covariance=True)

        # Depth maps
        chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.NoFiltering)
        doc.save()

        # Dense & products
        has_transform = (chunk.transform is not None) and (chunk.transform.matrix is not None)
        if has_transform:
            chunk.buildDenseCloud(point_confidence=True)
            doc.save()

            if chunk.dense_cloud:
                # Remove low-confidence points
                minconf = 3
                chunk.dense_cloud.setConfidenceFilter(0, minconf - 1)
                all_classes = list(range(128))
                chunk.dense_cloud.removePoints(all_classes)  # removes currently active (low-conf) points
                chunk.dense_cloud.resetFilters()
                doc.save()

                chunk.buildDem(source_data=Metashape.DenseCloudData)
                doc.save()

                chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
                doc.save()

        # Exports
        target_crs = Metashape.CoordinateSystem('EPSG::32715')
        projection = Metashape.OrthoProjection()
        projection.crs = target_crs

        chunk_folder = os.path.join(output_folder, chunk.label)
        os.makedirs(chunk_folder, exist_ok=True)

        chunk.exportReport(path=os.path.join(chunk_folder, f"{chunk.label}_report.pdf"))

        if chunk.model:
            chunk.exportModel(crs=target_crs, path=os.path.join(chunk_folder, f"{chunk.label}_model.obj"))

        if chunk.dense_cloud:
            chunk.exportPoints(
                crs=target_crs,
                path=os.path.join(chunk_folder, f"{chunk.label}.las"),
                source_data=Metashape.DenseCloudData,
            )

        if chunk.elevation:
            chunk.exportRaster(
                projection=projection,
                path=os.path.join(chunk_folder, f"{chunk.label}_dem.tif"),
                source_data=Metashape.ElevationData,
            )

        if chunk.orthomosaic:
            chunk.exportRaster(
                projection=projection,
                path=os.path.join(chunk_folder, f"{chunk.label}.tif"),
                source_data=Metashape.OrthomosaicData,
            )

        if chunk.cameras:
            chunk.exportCameras(
                crs=target_crs,
                path=os.path.join(chunk_folder, f"{chunk.label}_cams.xml"),
                format=Metashape.CamerasFormatXML,
            )

        logfile.write(f"Processing of {chunk.label} was successful.\n")

    logfile.write(time.strftime("Processing finished at %Y-%m-%d %H:%M:%S\n"))
    logfile.close()
    print(f"Processing finished, results saved to {output_folder}.")


if __name__ == "__main__":
    main()