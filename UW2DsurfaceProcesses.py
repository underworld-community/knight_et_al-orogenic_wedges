from __future__ import print_function,  absolute_import
import abc
import underworld.function as fn
import numpy as np
import math
import sys
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata, interp1d, InterpolatedUnivariateSpline
from UWGeodynamics import non_dimensionalise as nd
from UWGeodynamics import dimensionalise
from UWGeodynamics import UnitRegistry as u
from mpi4py import MPI as _MPI
from tempfile import gettempdir

### version check for the surface tracers
from UWGeodynamics import __version__ as version

comm = _MPI.COMM_WORLD
rank = comm.rank
size = comm.size

ABC = abc.ABCMeta('ABC', (object,), {})
_tempdir = gettempdir()


class SurfaceProcesses(ABC):

    def __init__(self, Model=None):

        self.Model = Model

    @property
    def Model(self):
        return self._Model

    @Model.setter
    def Model(self, value):
        self._Model = value
        if value:
            self._init_model()

    @abc.abstractmethod
    def _init_model(self):
        pass

    @abc.abstractmethod
    def solve(self, dt):
        pass


class diffusiveSurfaceErosion(SurfaceProcesses):
    """Linear diffusion surface
    """


    def __init__(self, airIndex, sedimentIndex, D, dx, surfaceElevation=0., Model=None, timeField=None):
        """
        Parameters
        ----------

            airIndex :
                air index
            sedimentIndex :
                sediment Index
            D :
                Diffusive rate, in unit length^2 / unit time
            dx :
                resolution of the surface, in unit length
            surfaceElevation:
                boundary between air and crust/material, unit length of y coord to be given

            ***All units are converted under the hood***


            ***
            usage:
            Model.surfaceProcesses = diffusiveSurfaceErosion(
                airIndex=air.index,
                sedimentIndex=Sediment.index,
                D= 100.0*u.meter**2/u.year,
                dx=5 * u.kilometer,
                surfaceElevation=0.0*u.kilometer
            )
            ***
        """


        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.timeField = timeField
        self.dx = dx.to(u.kilometer).magnitude
        self.surfaceElevation = surfaceElevation.to(u.kilometer).magnitude
        self.D  = D.to(u.kilometer**2 / u.year).magnitude
        self.Model = Model
        self.surfaceTracers = None
        self.surface_dt_diffusion = (0.2 * (self.dx * self.dx / self.D))

        self.surface_data_local = None

    def _init_model(self):
        '''set up custom tracers for surface'''
        x_min_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,0].min()
        x_max_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,0].max()

        '''creates initial surface on each node'''
        x_surf = np.arange(x_min_local,x_max_local,nd(self.dx * u.kilometer))

        ''' create surface tracers to advect on each node'''
        self.surface_data_local = np.zeros((x_surf.shape[0], 2), dtype='float64')

        self.surface_data_local[:,0] = x_surf
        self.surface_data_local[:,1] = nd(self.surfaceElevation * u.kilometer)


        ''' output surface tracers  '''
        npoints = int(self.Model.maxCoord[0].to(u.kilometer).magnitude/self.dx)
        coords = np.ndarray((npoints, 2))
        coords[:, 0] = np.linspace(nd(self.Model.minCoord[0]), nd(self.Model.maxCoord[0]), npoints)
        coords[:, 1] = nd(self.surfaceElevation * u.kilometer)
        ### Docker version is 2.9.5
        if version < '2.9.6':
            self.surfaceTracers = self.Model.add_passive_tracers(name="surfaceTracers", vertices=[coords[:, 0], coords[:, 1].min()], zOnly=True)
        else:
            self.surfaceTracers = self.Model.add_passive_tracers(name="surfaceTracers", vertices=coords, advect=False)


    def solve(self, dt):

        root_proc = 0

        z_max_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,1].max()
        z_min_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,1].min()

        ### collect surface data on each node
        if z_max_local >= self.surface_data_local[:,1].min():
            ### gets the x and z data of the surface tracers
            x_data = np.ascontiguousarray(self.surface_data_local[:,0].copy())
            z_data = np.ascontiguousarray(self.surface_data_local[:,1].copy())

            ### Get the velocity of the surface tracers
            tracer_velocity = self.Model.velocityField.evaluate(self.surface_data_local)
            vx = np.ascontiguousarray(tracer_velocity[:,0])
            vz = np.ascontiguousarray(tracer_velocity[:,1])
        else:
            ### creates dummy data on nodes without the surface
            x_data = np.array([None], dtype='float64')
            z_data = np.array([None], dtype='float64')
            vx = np.array([None], dtype='float64')
            vz = np.array([None], dtype='float64')



        ### Collect local array sizes using the high-level mpi4py gather
        sendcounts = np.array(comm.gather(len(x_data), root=root_proc))

        comm.barrier()

        if rank == root_proc:
        ### creates dummy data on all nodes to store the surface
            # surface_data = np.zeros((npoints,2))
            x_surface_data = np.zeros((sum(sendcounts)), dtype='float64')
            z_surface_data = np.zeros((sum(sendcounts)), dtype='float64')
            vx_data = np.zeros((sum(sendcounts)), dtype='float64')
            vz_data = np.zeros((sum(sendcounts)), dtype='float64')
            surface_data = np.zeros((sum(sendcounts), 4), dtype='float64')
        else:
            x_surface_data = None
            z_surface_data = None
            vx_data = None
            vz_data = None
            surface_data = None

        ### store the surface spline on each node
        f1 = None

        comm.barrier()

        ## gather x values, can't do them together
        comm.Gatherv(sendbuf=x_data, recvbuf=(x_surface_data, sendcounts), root=root_proc)
        ## gather z values
        comm.Gatherv(sendbuf=z_data, recvbuf=(z_surface_data, sendcounts), root=root_proc)

        ### gather velocity values
        comm.Gatherv(sendbuf=vx, recvbuf=(vx_data, sendcounts), root=root_proc)

        comm.Gatherv(sendbuf=vz, recvbuf=(vz_data, sendcounts), root=root_proc)



        if rank == root_proc:
            ### Put back into combined array
            surface_data[:,0] = x_surface_data
            surface_data[:,1] = z_surface_data
            surface_data[:,2] = vx_data
            surface_data[:,3] = vz_data

            ### remove dummy data
            surface_data = surface_data[~np.isnan(surface_data[:,0])]

            ### sort by x values
            surface_data = surface_data[np.argsort(surface_data[:, 0])]

            # # Advect top surface
            x2 = surface_data[:,0] + (surface_data[:,2] * dt)
            z2 = surface_data[:,1] + (surface_data[:,3] * dt)


            # # Spline top surface
            f = interp1d(x2, z2, kind='cubic', fill_value='extrapolate')

            ### update surface tracer position
            # surface_data[:,0] = (surface_data[:,0])
            surface_data[:,1] = f(surface_data[:,0])


            ### gets the x and y coordinates from the tracers
            x = dimensionalise(surface_data[:,0], u.kilometer).magnitude
            z = dimensionalise(surface_data[:,1], u.kilometer).magnitude


            ### time to diffuse surface based on Model dt
            total_time = (dimensionalise(dt, u.year)).magnitude

            '''Diffusion surface process'''
            '''erosion dt for diffusion surface'''
            dt = min(self.surface_dt_diffusion, total_time)

            nts = math.ceil(total_time/dt)

            dt = total_time / nts
            print('total time:', total_time, 'timestep:', dt, 'No. of its:', nts, flush=True)

            z_orig = z.copy()

            ### Basic Hillslope diffusion
            for i in range(nts):
                qs = -self.D * np.diff(z)/self.dx
                dzdt = -np.diff(qs)/self.dx
                z[1:-1] += dzdt*dt


            x_nd = nd(x*u.kilometer)

            z_nd = nd(z*u.kilometer)


            ''' creates a no slip boundary condition for the surface'''
            z_nd[0] = nd(self.surfaceElevation * u.kilometer)
            z_nd[-1] = nd(self.surfaceElevation * u.kilometer)
            ### creates function for the new surface that has eroded, to be broadcast back to nodes

            f1 = interp1d(x_nd, z_nd, fill_value='extrapolate', kind='cubic')
            # f1 = InterpolatedUnivariateSpline(x_nd, z_nd, k=1.)


            print('finished surface process on global rank:', rank, flush= True)

        comm.barrier()

        '''broadcast the new surface'''
        ### broadcast function for the surface
        f1 = comm.bcast(f1, root=root_proc)

        comm.barrier()

        ''' replaces the new diffused surface data, only changes z as x values don't change '''
        ### update the surface on individual nodes
        self.surface_data_local[:,1] = f1(self.surface_data_local[:,0])
        ### update the global surface tracers
        self.surfaceTracers.data[:,1] = f1(self.surfaceTracers.data[:,0])

        '''Erode surface/deposit sed based on the surface'''
        ### update the material on each node according to the spline function for the surface
        self.Model.materialField.data[(self.Model.swarm.data[:,1] > f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] != self.airIndex)] = self.airIndex
        self.Model.materialField.data[(self.Model.swarm.data[:,1] < f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] == self.airIndex)] = self.sedimentIndex


        comm.barrier()

        return

class velocitySurfaceErosion(SurfaceProcesses):
    """velocity surface erosion
    """


    def __init__(self, airIndex, sedimentIndex, sedimentationRate, erosionRate, dx, surfaceElevation=0., Model=None, timeField=None):
        """
        Parameters
        ----------

            airIndex :
                air index
            sedimentIndex :
                sediment Index
            sedimentaitonRate :
                Rate at which to deposit sediment, in unit length / unit time
            erosionRate :
                Rate at which to erode surface, in unit length / unit time
            surfaceElevation:
                boundary between air and crust/material, unit length of y coord to be given

            ***All units are converted under the hood***

            ***
            usage:
            Model.surfaceProcesses = velocitySurfaceErosion(
                airIndex=air.index,
                sedimentIndex=Sediment.index,
                sedimentationRate= 1.0*u.millimeter / u.year,
                erosionRate= 0.5*u.millimeter / u.year,
                dx=5 * u.kilometer,
                surfaceElevation=0.0*u.kilometer
            )
            ***

        """


        self.airIndex = airIndex
        self.sedimentIndex = sedimentIndex
        self.timeField = timeField
        self.dx = dx.to(u.kilometer).magnitude
        self.surfaceElevation = surfaceElevation.to(u.kilometer).magnitude
        self.sedimentationRate  = sedimentationRate.to(u.kilometer / u.year).magnitude
        self.erosionRate  = -1. * erosionRate.to(u.kilometer / u.year).magnitude
        self.Model = Model
        self.surfaceTracers = None

        self.surface_data_local = None

    def _init_model(self):
        '''set up custom tracers for surface'''
        x_min_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,0].min()
        x_max_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,0].max()

        '''creates initial surface on each node'''
        x_surf = np.arange(x_min_local,x_max_local,nd(self.dx * u.kilometer))

        ''' create surface tracers to advect on each node'''
        self.surface_data_local = np.zeros((x_surf.shape[0], 2), dtype='float64')

        self.surface_data_local[:,0] = x_surf
        self.surface_data_local[:,1] = nd(self.surfaceElevation * u.kilometer)


        ''' output surface tracers  '''
        npoints = int(self.Model.maxCoord[0].to(u.kilometer).magnitude/self.dx)
        coords = np.ndarray((npoints, 2))
        coords[:, 0] = np.linspace(nd(self.Model.minCoord[0]), nd(self.Model.maxCoord[0]), npoints)
        coords[:, 1] = nd(self.surfaceElevation * u.kilometer)

        if version < '2.9.6':
            self.surfaceTracers = self.Model.add_passive_tracers(name="SurfaceTracers", vertices=[coords[:, 0], coords[:, 1].min()], zOnly=True)
        else:
            self.surfaceTracers = self.Model.add_passive_tracers(name="SurfaceTracers", vertices=coords, advect=False)

    def solve(self, dt):

        root_proc = 0

        z_max_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,1].max()
        z_min_local = self.Model.mesh.data[:self.Model.mesh.nodesLocal,1].min()

        ### collect surface data on each node
        if z_max_local >= self.surface_data_local[:,1].min():
            ### gets the x and z data of the surface tracers
            x_data = np.ascontiguousarray(self.surface_data_local[:,0].copy())
            z_data = np.ascontiguousarray(self.surface_data_local[:,1].copy())

            ### Get the velocity of the surface tracers
            tracer_velocity = self.Model.velocityField.evaluate(self.surface_data_local)
            vx = np.ascontiguousarray(tracer_velocity[:,0])
            vz = np.ascontiguousarray(tracer_velocity[:,1])
        else:
            ### creates dummy data on nodes without the surface
            x_data = np.array([None], dtype='float64')
            z_data = np.array([None], dtype='float64')
            vx = np.array([None], dtype='float64')
            vz = np.array([None], dtype='float64')



        ### Collect local array sizes using the high-level mpi4py gather
        sendcounts = np.array(comm.gather(len(x_data), root=root_proc))

        comm.barrier()

        if rank == root_proc:
        ### creates dummy data on all nodes to store the surface
            # surface_data = np.zeros((npoints,2))
            x_surface_data = np.zeros((sum(sendcounts)), dtype='float64')
            z_surface_data = np.zeros((sum(sendcounts)), dtype='float64')
            vx_data = np.zeros((sum(sendcounts)), dtype='float64')
            vz_data = np.zeros((sum(sendcounts)), dtype='float64')
            surface_data = np.zeros((sum(sendcounts), 4), dtype='float64')
        else:
            x_surface_data = None
            z_surface_data = None
            vx_data = None
            vz_data = None
            surface_data = None

        ### store the surface spline on each node
        f1 = None

        comm.barrier()

        ## gather x values, can't do them together
        comm.Gatherv(sendbuf=x_data, recvbuf=(x_surface_data, sendcounts), root=root_proc)
        ## gather z values
        comm.Gatherv(sendbuf=z_data, recvbuf=(z_surface_data, sendcounts), root=root_proc)

        ### gather velocity values
        comm.Gatherv(sendbuf=vx, recvbuf=(vx_data, sendcounts), root=root_proc)

        comm.Gatherv(sendbuf=vz, recvbuf=(vz_data, sendcounts), root=root_proc)



        if rank == root_proc:
            ### Put back into combined array
            surface_data[:,0] = x_surface_data
            surface_data[:,1] = z_surface_data
            surface_data[:,2] = vx_data
            surface_data[:,3] = vz_data

            ### remove dummy data
            surface_data = surface_data[~np.isnan(surface_data[:,0])]

            ### sort by x values
            surface_data = surface_data[np.argsort(surface_data[:, 0])]

            # # Advect top surface
            x2 = surface_data[:,0] + (surface_data[:,2] * dt)
            z2 = surface_data[:,1] + (surface_data[:,3] * dt)


            # # Spline top surface
            f = interp1d(x2, z2, kind='cubic', fill_value='extrapolate')

            ### update surface tracer position
            # surface_data[:,0] = (surface_data[:,0])
            surface_data[:,1] = f(surface_data[:,0])


            ### gets the x and y coordinates from the tracers
            x = dimensionalise(surface_data[:,0], u.kilometer).magnitude
            z = dimensionalise(surface_data[:,1], u.kilometer).magnitude


            ### time to diffuse surface based on Model dt
            total_time = (dimensionalise(dt, u.year)).magnitude

            '''Velocity surface process'''

            '''erosion dt for vel model'''

            Vel_for_surface = max(abs(self.erosionRate * u.kilometer / u.year),abs(self.sedimentationRate*u.kilometer / u.year), abs(dimensionalise(self.Model.velocityField.data.max(), u.kilometer/u.year)))


            surface_dt_vel = (0.2 *  (self.dx / Vel_for_surface.magnitude))

            dt = min(surface_dt_vel, total_time)

            nts = math.ceil(total_time/dt)
            dt = total_time / nts

            print('total time:', total_time, 'timestep:', dt, 'No. of its:', nts, flush=True)

            z_orig = z.copy()

            ### Velocity erosion/sedimentation rates for the surface
            for i in range(nts):
                Ve_loop = np.where(z <= 0., 0., self.erosionRate)
                Vs_loop = np.where(z >= 0., 0., self.sedimentationRate)

                dzdt =  Vs_loop + Ve_loop

                z[:] += dzdt*dt


            x_nd = nd(x*u.kilometer)

            z_nd = nd(z*u.kilometer)

            ''' creates a no slip boundary condition for the surface'''
            z_nd[0] = nd(self.surfaceElevation * u.kilometer)
            z_nd[-1] = nd(self.surfaceElevation * u.kilometer)

            ### creates function for the new surface that has eroded, to be broadcast back to nodes

            f1 = interp1d(x_nd, z_nd, fill_value='extrapolate', kind='cubic')
            # f1 = InterpolatedUnivariateSpline(x_nd, z_nd, k=1.)


            print('finished surface process on global rank:', rank, flush= True)

        comm.barrier()

        '''broadcast the new surface'''
        ### broadcast function for the surface
        f1 = comm.bcast(f1, root=root_proc)

        comm.barrier()

        ''' replaces the new diffused surface data, only changes z as x values don't change '''
        ### update the surface on individual nodes
        self.surface_data_local[:,1] = f1(self.surface_data_local[:,0])
        ### update the global surface tracers
        self.surfaceTracers.data[:,1] = f1(self.surfaceTracers.data[:,0])

        '''Erode surface/deposit sed based on the surface'''
        ### update the material on each node according to the spline function for the surface
        self.Model.materialField.data[(self.Model.swarm.data[:,1] > f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] != self.airIndex)] = self.airIndex
        self.Model.materialField.data[(self.Model.swarm.data[:,1] < f1(self.Model.swarm.data[:,0])) & (self.Model.materialField.data[:,0] == self.airIndex)] = self.sedimentIndex


        comm.barrier()

        return
