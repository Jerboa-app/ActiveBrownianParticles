# This code uses GLMakie (so GPU enabled with OpenGL)
# The code should work on CPU if Makie is substituted for
# GLMakie, but will be slower
using DelimitedFiles, ArgParse, GLMakie, PerceptualColourMaps, ProgressMeter, Distances
# this will tell GLMakie to go as fast as possible
GLMakie.set_window_config!(framerate = Inf, vsync = false)
# this will suppress the plot window = faster plots + does not takeover your pc
# trying to plot and display!
GLMakie.inline!(true)

include("include/AlphaShapes/AlphaShapes.jl")

import ArgParse.parse_item

function parse_item(::Type{Array{Float64,1}}, x::AbstractString)
    """overload to parse array types using ArgParse

        e.g specify an argument like so

        \"--my-array-argument\"
            arg_type = Array{Float64,1}
            default = [1.0]

        and then the code can be run like

        julia foo.jl --my-array-argument [1.0,3.14,1,100.0]
        """
    return parse.(Float64,split(x[2:end-1],","))
end

function parse_item(::Type{Union{Int, Float64}}, x::AbstractString)
    """overload to parse Union{Int, Float64}
    """
    return parse.(Int,x)
end


s = ArgParseSettings()

@add_arg_table! s begin
	"--path"
	help = "output path"
	arg_type = String
	default = "out.mp4"

	"--upscaling"
	help = "x times increase to 1920x1080 output resolution"
	arg_type = Int
	default = 1

	"--framerate"
	help = "frame rate of video"
	arg_type = Int
	default = 60

	"--colour"
	help = "colour by density"
	arg_type = Bool
	default = true

	"--trajectories"
	help = "path to trajectory data"
	arg_type = String
	default = "trajectories.txt"

	"--log-colour"
	help = "colour scale is log10"
	arg_type = Bool
	default = false

	"--radius"
	help = "particle radius"
	arg_type = Float64
	default = 1.0

	"--ellipse"
	help = "minor axis length, 0 plots a circle"
	arg_type = Float64
	default = 0.0

	"--periodic-box-length"
	help = "length of periodic box, 0 = not periodic"
	arg_type = Float64
	default = 0.0

	"--bounds"
	help = "manually set axis limits"
	arg_type = Array{Float64,1}
	default = [-1.,-1.,-1.,-1.]

	"--trails"
	help = "plot (fading) particle trails of this length"
	arg_type = Int
	default = 0

	"--arrows"
	help = "plot arrows for particle heading, values > 0 increase arrow length"
	arg_type = Float64
	default = 0.0

	"--max-time"
	help = "number of frames"
	arg_type = Union{Int,Float64}
	default = Inf

	"--flashing-bounds"
	help = "flash a colour when a particle passes a boundary (periodic)"
	arg_type = Int
	default = 0

	"--offset"
	help = "timestep offset"
	arg_type = Int
	default = 1

	"--sizes"
	help = "file with particle sizes"
	arg_type = String
	default = ""
end

args = parse_args(s)

@info args

upscale = args["upscaling"]
path = args["path"]
fps = args["framerate"]
data = args["trajectories"]
radius = args["radius"]
logcolour = args["log-colour"]
bounds = args["bounds"]
trails = args["trails"]
periodic = args["periodic-box-length"]
arrows = args["arrows"]
T_max = args["max-time"]
flash = args["flashing-bounds"]
colour = args["colour"]
offset = args["offset"]
ellipse = args["ellipse"]
sizes = args["sizes"]

"""
Determines number of particles and reads the trajectory
data into a time X particle_index X dimension tensor
"""
function ReadData(PATH)
    Data = readdlm("$PATH",',')
    file = open(PATH,"r")
    N = 0
    while eof(file) == false
        l = readline(file)
        if l == ""
	    break
        end
	N += 1
    end
    close(file)
    @info "Reading N = $N Particles, from file $PATH"
    t_max = Int(size(Data,1) / N)
    T = zeros(t_max,N,4)
    t = 1
    j = 1
    for i in 1:size(Data,1)
        T[t,j,:] = Data[i,:]
        if i % N == 0
            t += 1
            j = 1
        else
            j += 1
        end
    end
    return T
end

"""
	ReadData, but up to a maximum timestep, also reads the file line by line
	which is useful for too large files (piecewise rendering), but is slower
"""
function ReadData(PATH,t_max)
    file = open(PATH,"r")
    N = 0
    while eof(file) == false
        l = readline(file)
        if l == ""
	    break
        end
	N += 1
    end
    close(file)
    @info "Reading N = $N Particles, from file $PATH"
    t_max == nothing ? t_max = Int(size(Data,1) / N) : nothing
    T = zeros(t_max,N,4)
    t = 1
    j = 1
	file = open(PATH,"r")
		prog = Progress(t_max)
    while (t <= t_max && eof(file) == false)
		data = readline(file)
		if data != ""
			T[t,j,:] = parse.(Float64,split(data,", "))
			j += 1
		else
			# new time step
			t += 1
			j = 1
			next!(prog)
		end
    end
    return T
end

"""
A delaunay triangulation based local density measure
"""
function Density(X)
	if periodic == 0
	    tess,inds = AlphaShapes.GetDelaunayTriangulation(X,true);
		d = zeros(size(X,1))
		for i in 1:size(X,1)
	    	d[i] = AlphaShapes.WeightedDTFELocalDensity(i,tess,inds)
		end
		return d
	end

    d = zeros(size(X,1))
    D = pairwise(PeriodicEuclidean([periodic,periodic]),X')
    # for i in 1:size(X,1)
    #     for j in 1:size(X,1)
    #         Rx = X[j,1]-X[i,1]
    #         Ry = X[j,2]-X[i,2]
    #         if (Rx > periodic*0.5)
    #             Rx -= periodic
    #         elseif (Rx <= -periodic*0.5)
    #             Rx += periodic
    #         end
	#
    #         if (Ry > periodic*0.5)
    #             Ry -= periodic
    #         elseif (Ry <= -periodic*0.5)
    #             Ry += periodic
    #         end
    #         D[i,j] = Rx*Rx+Ry*Ry
    #     end
    # end

	for i in 1:size(X,1)
        nn = D[i,:][D[i,:].>0]
		d[i] = 1.0/ (sum(sort(nn)[1:7])/7.)^2.0
	end

	return d

end

cm = cmap("heat")

"""
Calculates local density at each particle and spits out a colour
on the given scale (colour_bins), a is the particle length
"""
function GetColours(T,i,colour_bins,a=2.0)
	if colour == false
			return [RGBAf0(0.,0.,1.,.173) for j in 1:size(T,2)]
	end
	if size(T,2) < 4
		if size(T,2) == 2
			return [RGBAf0(0.,0.,1.,1.),RGBAf0(1.,0.,0.,.1)]
		else
			return [RGBAf0(0.,0.,1.,.173) for j in 1:size(T,2)]
		end
	end
    ρ = Density(T[i,:,1:2]./(2.0*a))
    c = Vector{PerceptualColourMaps.RGBA{Float64}}()
    for j in 1:size(T,2)
        index = findfirst(x->x.>=ρ[j],colour_bins)
        if index == nothing
            index = length(colour_bins)
        end
        push!(c,cm[index])
    end
    return c
end

if isinf(T_max)
	T = ReadData(data)
	T_max = size(T,1)
else
	T = ReadData(data,T_max)
end


if size(T,2) < 4
	@warn "Density requires 4 or more particles, colour will be constant"
end

fig = Figure(resolution=(upscale*1920,upscale*1080))
ax = Axis(fig[1,1])

# for time keeping
time_ = Node(1)

# GLMakie is a really a code base around OpenGL
# so nodes are a way to tell makie we have a geometry
# in this case a point that we can update as the time node
# advances. This is much faster than calling scatter! or lines!
# each iteration of the record loop below, since GLMakie can
# pre allocate vertex buffers etc in OpenGL and
# alter them in an optimised way as time progresses

# store positions in a node vector
particles = Node([Point2(T[offset,i,1:2]...) for i in 1:size(T,2)])
if periodic > 0
	mirror_particles = Node([Point2(NaN,NaN) for i in 1:size(T,2)*4])
end
# same for orientations
θ = Node([T[1,i,3] for i in 1:size(T,2)])

# if logcolour
#     colour_bins = 10.0.^collect(range(-3.0,stop=1.0,length=256))
# else
#     colour_bins = collect(range(1e-3,stop=6.0,length=256))
# end

if logcolour
    colour_bins = 10.0.^collect(range(-2.0,stop=1.0,length=256))
else
    colour_bins = collect(range(1e-1,stop=2.0,length=256))
end

# now a node for colours (densities here)
colours = Node(GetColours(T,offset,colour_bins,1))
if periodic > 0
	mirror_colours = Node([colours[][i] for i in 1:size(T,2) for j in 1:4])
end
# @show colours
# @show mirror_colours

# Hx, Hy, and A are more complicated
# These are "histories" of particle positions for trails
# A are colours, N*(2*trails+2) of them, and Hx and Hy are positions
# these will be passed to GLMakie.linesegments! later which draw
# a line segment between pairs of points (x1,x2) and (y1,y2) etc.
# Why do this instead of GLMakie.lines! ? it seems then I could not
# have a variable Colour across bits of the same line.
# NaN points can be used to split segments up, so it looks
# like we have drawn many GLMakie.line! 's
Hx = Vector{Float64}()
Hy = Vector{Float64}()
A = Vector{RGBAf0}()

if trails > 1
	for j in 1:size(T,2)
		for i in 1:trails
			push!(Hx,NaN)
			push!(Hx,NaN)
			push!(Hy,NaN)
			push!(Hy,NaN)
			push!(A,RGBAf0(0.,0.,0.,0.))
		end
		# these NaN's will be kept to separate line segments
		push!(Hx,NaN)
		push!(Hx,NaN)
		push!(Hy,NaN)
		push!(Hy,NaN)
		push!(A,RGBAf0(0.,0.,0.,0.))
	end
end

Hx = Node(Hx)
Hy = Node(Hy)
A = Node(A)
# v is the alpha decay exponent (see later)
v = -0.25
ABs = Vector{Vec2f0}()

if (sizes!="")
	s = readdlm(sizes,',')
	[push!(ABs,Vec2f0(2.0.*s[i,:])) for i in 1:size(s,1)]
else
	if ellipse > 0
		[push!(ABs,Vec2f0(2*radius,2*ellipse)) for i in 1:size(T,2)]
	else
		[push!(ABs,Vec2f0(2*radius,2*radius)) for i in 1:size(T,2)]
	end
end



# we draw all our nodes so GLMakie init's them and the figure

scatter!(ax,particles,
    markersize = ABs,
    markerspace=SceneSpace,
    rotations=θ,
    color=colours,
	strokewidth=1.0
)

dir = Node([Point2(NaN,NaN) for i in 1:size(T,2)])
mirror_dir = Node([Point2(NaN,NaN) for i in 1:size(T,2) for j in 1:4])
mθ = Node([NaN for i in 1:size(T,2) for j in 1:4])
arrow_length = radius*2+arrows
if arrows > 0
	arrows!(ax,particles,dir,lengthscale=radius*2,linewidth=4.,arrowsize=arrows,markerspace=SceneSpace)
end

flash_lines = Node([Point2(NaN,NaN),Point2(NaN,NaN),Point2(NaN,NaN),Point2(NaN,NaN)]) # up to 2 at once
flash_colours = Node([RGBAf0(1.,0.,0.,1.),RGBAf0(0.,0.,1.,1.)])

if periodic > 0
	if ellipse > 0.0
		scatter!(ax,mirror_particles,
		    markersize = (2*radius,2*ellipse),
		    markerspace=SceneSpace,
			rotations=mθ,
			color=mirror_colours,
			strokewidth=1.0
		)
	else
		scatter!(ax,mirror_particles,
		    markersize = (2*radius,2*radius),
		    markerspace=SceneSpace,
			rotations=mθ,
			color=mirror_colours,
			strokewidth=1.0
		)
	end
	if arrows > 0
		arrows!(ax,mirror_particles,mirror_dir,lengthscale=radius*2,linewidth=4.,arrowsize=arrows,markerspace=SceneSpace)
	end

	if flash > 0
		linesegments!(ax,flash_lines,color=flash_colours,linewidth=30)
	end
end

if trails > 1
	# draw historys (but they are just NaN here and alpha=0)
	GLMakie.linesegments!(ax,Hx,Hy,color=A,linewidth=4.0)
end

if bounds == [-1.,-1.,-1.,-1.]
    bounds = [minimum(T[:,:,1])-radius,maximum(T[:,:,1])+radius,minimum(T[:,:,2])-radius,maximum(T[:,:,2])+radius]
end

limits!(ax,bounds...)
# aspect 1 so geometries are as they should be without needing projection matrices
ax.aspect = AxisAspect(1)
fig

render_arrows = arrows > 0

l = 0
flash_flag = false

@info "Beginning rendering shortly..."
prog = Progress(Int(floor(T_max))-offset-1)
record(fig, path, collect((1+offset):1:size(T,1)), framerate=fps) do i
	time_[] = i
    new_particles = particles[]
	new_dir = dir[]
	new_mirror_dir = mirror_dir[]
	if periodic > 0
		new_mirror_particles = mirror_particles[]
		new_mirror_colours = mirror_colours[]
	end
    new_θ = θ[]
	new_mθ = mθ[]
	new_Hx = Hx[]
	new_Hy = Hy[]
	new_A = A[]
	new_colours = GetColours(T,i,colour_bins,1.)
    for j in 1:length(new_particles)
		# the beauty of GLMakie, just update the nodes
        new_particles[j] = Point2(T[i,j,1:2]...)
        new_θ[j] = T[i,j,3]
		if render_arrows
			new_dir[j] = Point2(cos(T[i,j,3]),sin(T[i,j,3]))
		end

		if periodic > 0
			new_mirror_particles[(j-1)*4+1] = Point2(NaN,NaN)
			new_mirror_particles[(j-1)*4+2] = Point2(NaN,NaN)
			new_mirror_particles[(j-1)*4+3] = Point2(NaN,NaN)
			new_mirror_particles[(j-1)*4+4] = Point2(NaN,NaN)
			new_mirror_dir[(j-1)*4+1] = Point2(cos(T[i,j,3]),sin(T[i,j,3]))
			new_mθ[(j-1)*4+1]  = T[i,j,3]
			new_mirror_dir[(j-1)*4+2] = Point2(cos(T[i,j,3]),sin(T[i,j,3]))
			new_mθ[(j-1)*4+2]  = T[i,j,3]
			new_mirror_dir[(j-1)*4+3] = Point2(cos(T[i,j,3]),sin(T[i,j,3]))
			new_mθ[(j-1)*4+3]  = T[i,j,3]
			new_mirror_dir[(j-1)*4+4] = Point2(cos(T[i,j,3]),sin(T[i,j,3]))
			new_mθ[(j-1)*4+4]  = T[i,j,3]

			# flash boundaries
			if flash > 0
				if (T[i,j,1] < radius && T[i,j,2] < radius)
					flash_lines[] = [Point2(0.0,0.0),Point2(periodic,0.0),Point2(0.0,0.0),Point2(0.0,periodic)]
					global flash_flag = true
					global l = 0
				elseif (T[i,j,1] > periodic-radius && T[i,j,2] > periodic-radius)
					flash_lines[] = [Point2(periodic,0.0),Point2(periodic,periodic),Point2(0.0,periodic),Point2(periodic,periodic)]
					global flash_flag = true
					global l = 0
				elseif (T[i,j,1] < radius && T[i,j,2] > periodic-radius)
					flash_lines[] = [Point2(0.0,0.0),Point2(0.0,periodic),Point2(0.0,periodic),Point2(periodic,periodic)]
					global flash_flag = true
					global l = 0
				elseif (T[i,j,1] > periodic-radius && T[i,j,2] < radius)
					flash_lines[] = [Point2(periodic,0.0),Point2(periodic,periodic),Point2(0.0,0.0),Point2(periodic,0.0)]
					global flash_flag = true
					global l = 0
				else
					# not a corner
					if (T[i,j,1] < radius)
						flash_lines[] = [Point2(0.0,0.0),Point2(0.0,periodic),Point2(NaN,NaN),Point2(NaN,NaN)]
						global flash_flag = true
						global l = 0
					elseif (T[i,j,1] > periodic-radius)
						flash_lines[] = [Point2(periodic,0.0),Point2(periodic,periodic),Point2(NaN,NaN),Point2(NaN,NaN)]
						global flash_flag = true
						global l = 0
					end

					if (T[i,j,2] < radius)
						flash_lines[] = [Point2(NaN,NaN),Point2(NaN,NaN),Point2(0.0,0.0),Point2(periodic,0.0)]
						global flash_flag = true
						global l = 0
					elseif (T[i,j,2] > periodic-radius)
						flash_lines[] = [Point2(NaN,NaN),Point2(NaN,NaN),Point2(0.0,periodic),Point2(periodic,periodic)]
						global flash_flag = true
						global l = 0
					end
				end

				if flash_flag && l < flash
					global l += 1
					flash_colours[] = [RGBAf0(1.,0.,0.,1.0-l/flash),RGBAf0(0.,0.,1.,1.0-l/flash)]
				else
					global l = 0
					global flash_flag = false
					flash_lines[] = [Point2(NaN,NaN),Point2(NaN,NaN),Point2(NaN,NaN),Point2(NaN,NaN)]
				end
			end
			# mirror particles
			# corners
			if (T[i,j,1] < radius+arrow_length && T[i,j,2] < radius+arrow_length)
				new_mirror_particles[(j-1)*4+1] = Point2(periodic+T[i,j,1],periodic+T[i,j,2])
				new_mirror_particles[(j-1)*4+2] = Point2(T[i,j,1],T[i,j,2]+periodic)
				new_mirror_particles[(j-1)*4+3] = Point2(T[i,j,1]+periodic,T[i,j,2])
			elseif (T[i,j,1] > periodic-radius-arrow_length && T[i,j,2] > periodic-radius-arrow_length)
				new_mirror_particles[(j-1)*4+1] = Point2(T[i,j,1]-periodic,T[i,j,2]-periodic)
				new_mirror_particles[(j-1)*4+2] = Point2(T[i,j,1],T[i,j,2]-periodic)
				new_mirror_particles[(j-1)*4+3] = Point2(T[i,j,1]-periodic,T[i,j,2])
			elseif (T[i,j,1] < radius+arrow_length && T[i,j,2] > periodic-radius-arrow_length)
				new_mirror_particles[(j-1)*4+1] = Point2(periodic+T[i,j,1],T[i,j,2]-periodic)
				new_mirror_particles[(j-1)*4+2] = Point2(T[i,j,1],T[i,j,2]-periodic)
				new_mirror_particles[(j-1)*4+3] = Point2(T[i,j,1]+periodic,T[i,j,2])
			elseif (T[i,j,1] > periodic-radius-arrow_length && T[i,j,2] < radius+arrow_length)
				new_mirror_particles[(j-1)*4+1] = Point2(T[i,j,1]-periodic,periodic+T[i,j,2])
				new_mirror_particles[(j-1)*4+2] = Point2(T[i,j,1],T[i,j,2]+periodic)
				new_mirror_particles[(j-1)*4+3] = Point2(T[i,j,1]-periodic,T[i,j,2])
			else
				# not a corner
				flag = false
				if (T[i,j,1] < radius+arrow_length)
					flag = true
					new_mirror_particles[(j-1)*4+1] = Point2(periodic+T[i,j,1],T[i,j,2])
				elseif (T[i,j,1] > periodic-radius-arrow_length)
					flag = true
					new_mirror_particles[(j-1)*4+1] = Point2(T[i,j,1]-periodic,T[i,j,2])
				end

				if (T[i,j,2] < radius+arrow_length)
					if flag
						new_mirror_particles[(j-1)*4+1] = Point2(new_mirror_particles[j][1],T[i,j,2]+periodic)
					else
						new_mirror_particles[(j-1)*4+1] = Point2(T[i,j,1],T[i,j,2]+periodic)
					end
				elseif (T[i,j,2] > periodic-radius-arrow_length)
					if flag
						new_mirror_particles[(j-1)*4+1] = Point2(new_mirror_particles[j][1],T[i,j,2]-periodic)
					else
						new_mirror_particles[(j-1)*4+1] = Point2(T[i,j,1],T[i,j,2]-periodic)
					end
				end
			end
			for i in 1:4
				new_mirror_colours[(j-1)*4+i] = new_colours[j]
			end
		end

		for k in 1:trails
			# the trails are a little more complex,
			# but this is just funky flattened array indexing
			if (i-k) > 0
				x = T[(i-k),j,1]
				y = T[(i-k),j,2]
				new_Hx[(j-1)*(2*trails+2)+2*k] = x
				new_Hx[(j-1)*(2*trails+2)+2*k+1] = x
				new_Hy[(j-1)*(2*trails+2)+2*k] = y
				new_Hy[(j-1)*(2*trails+2)+2*k+1] = y
				fade = k^v # fade out based on how many timesteps in past
				new_A[(j-1)*(trails+1)+k] = RGBAf0(0.,0.,0.,1.0*fade)
			else
				new_A[(j-1)*(trails+1)+k] = RGBAf0(0.0,0.0,0.0,0.0)
			end
		end
    end
	# actually apply the node updates
    particles[] = new_particles
	if render_arrows
		dir[] = new_dir
	end
	if periodic > 0
		mirror_particles[] = new_mirror_particles
		mirror_colours[] = new_mirror_colours
		if render_arrows
			mirror_dir[] = new_mirror_dir
		end
		mθ[] = new_mθ
	end
    θ[] = new_θ
    colours[] = new_colours
	Hx[] = new_Hx
	Hy[] = new_Hy
	A[] = new_A
    next!(prog)
end
@info "Done! Saved movie to $path"
