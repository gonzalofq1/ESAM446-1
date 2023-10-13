import pytest
import numpy as np
import finite
import timesteppers

resolution_list = [100, 200, 400]

error_RK_2_2 = {100:0.5, 200:0.15, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_2_2(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference(grid)
    f = lambda u: d @ u

    steps = 2
    alpha = 0.4
    
    ts = timesteppers.AdamsBashforth(IC, f, steps, alpha*(grid.dx)**(4/3))

    
    num_periods = 1.8
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_RK_2_2[resolution]

    assert error < error_est

error_RK_2_4 = {100:0.15, 200:0.05, 400:0.01}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_2_4(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference4(grid)
    f = lambda u: d @ u

    steps = 2
    alpha = 0.4
    
    ts = timesteppers.AdamsBashforth(IC, f, steps, alpha*grid.dx**(4/3))


   
    num_periods = 1.2
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_RK_2_4[resolution]

    assert error < error_est

error_RK_3_2 = {100:0.5, 200:0.2, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_3_2(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference(grid)
    f = lambda u: d @ u

    steps = 3
    alpha = 0.4
    ts = timesteppers.AdamsBashforth(IC, f, steps, alpha*grid.dx**(4/3))

    
    num_periods = 1.8
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_RK_3_2[resolution]

    assert error < error_est

error_RK_3_4 = {100:0.04, 200:0.005, 400:3e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_3_4(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference4(grid)
    f = lambda u: d @ u

    steps = 3
    alpha = 0.25
    ts = timesteppers.AdamsBashforth(IC, f, steps, alpha*grid.dx**(4/3))


   
    num_periods = 1.2
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_RK_3_4[resolution]

    assert error < error_est

error_RK_4_2 = {100:0.5, 200:0.2, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_4_2(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference(grid)
    f = lambda u: d @ u
    
    steps = 4
    alpha = 0.25
    ts = timesteppers.AdamsBashforth(IC, f, steps, alpha*grid.dx**(4/3))


    
    num_periods = 1.8
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_RK_4_2[resolution]

    assert error < error_est

error_RK_4_4 = {100:0.04, 200:0.003, 400:2e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_RK_4_4(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference4(grid)
    f = lambda u: d @ u

    steps = 4
    alpha = 0.25
    ts = timesteppers.AdamsBashforth(IC, f, steps, alpha*grid.dx**(4/3))


   
    num_periods = 1.2
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_RK_4_4[resolution]

    assert error < error_est

