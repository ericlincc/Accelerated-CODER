# julia scripts/run_algo.jl <dataset> <algo> <Lipschitz> <gamma> (<K>)

using ArgParse
using CSV
using Dates
using LinearAlgebra
using Logging
using SparseArrays
using JLD2


BLAS.set_num_threads(1)


include("../src/problems/utils/data.jl")
include("../src/problems/utils/data_parsers.jl")
include("../src/problems/loss_func/logisticloss.jl")
include("../src/problems/reg_func/ridge.jl")
include("../src/problems/reg_func/elasticnet.jl")
include("../src/problems/composite_func.jl")

include("../src/algorithms/utils/exitcriterion.jl")
include("../src/algorithms/utils/results.jl")

# include("../src/algorithms/coder.jl")
include("../src/algorithms/acoder.jl")
# include("../src/algorithms/apcg.jl")
# include("../src/algorithms/rcdm.jl")
include("../src/algorithms/gd.jl")


# (d, n)
DATASET_INFO = Dict([
    ("sonar_scale", (60, 208)),
    ("a1a", (123, 1605)),
    ("a9a", (123, 32561)),
    ("gisette_scale", (5000, 6000)),
    ("news20", (1355191, 19996)),
    ("rcv1", (47236, 20242)),
    ("phishing", (68, 11055))
])


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--outputdir"
            help = "output directory"
            required = true
        "--maxiter"
            help = "max iterations"
            required = true
        "--maxtime"
            help = "max execution time in seconds"
            required = true
        "--targetaccuracy"
            help = "target accuracy"
            arg_type = Float64
            required = true
        "--optval"
            help = "known opt value"
            arg_type = Float64
            default = 0.0
        "--loggingfreq"
            help = "logging frequency"
            default = 100
        "--dataset"
            help = "choice of dataset"
            required = true
        "--lossfn"
            help = "choice of loss funcation"
            default = "logistic"
        "--lambda1"
            help = "elastic net lambda 1"
            arg_type = Float64
            default = 0.0
        "--lambda2"
            help = "elastic net lambda 2"
            arg_type = Float64
            default = 0.0
        "--algo"
            help = "algorithm to run"
            required = true
        "--lipschitz"
            help = "Lipschitz constant"
            arg_type = Float64
            required = true
        "--gamma"
            help = "gamma"
            arg_type = Float64
            default = 0.0
        "--K"
            help = "Variance reduction K"
            default = 0
    end
    return parse_args(s)
end


# Run setup
parsed_args = parse_commandline()
outputdir = parsed_args["outputdir"]
algorithm = parsed_args["algo"]


# Problem setup
dataset = parsed_args["dataset"]
elasticnet_λ₁ = parsed_args["lambda1"]
elasticnet_λ₂ = parsed_args["lambda2"]

d, n = DATASET_INFO[dataset]
if !haskey(DATASET_INFO, dataset)
    throw(ArgumentError("Invalid dataset name supplied."))
end
filepath = "../data/libsvm/$(dataset)"
data = libsvm_parser(filepath, n, d)
loss = LogisticLoss(data)
reg = ElasticNet(elasticnet_λ₁, elasticnet_λ₂)
problem = CompositeFunc(loss, reg)

@info "dataset = $(dataset); d = $(d); n = $(n)"
@info "elasticnet_λ₁ = $(elasticnet_λ₁); elasticnet_λ₂ = $(elasticnet_λ₂)"
@info "--------------------------------------------------"


# Exit criterion
maxiter = Int(parse(Float64, parsed_args["maxiter"]))
maxtime = Int(parse(Float64, parsed_args["maxtime"]))
targetaccuracy = parsed_args["targetaccuracy"] + parsed_args["optval"]
loggingfreq = parse(Int64, parsed_args["loggingfreq"])
exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)

@info "maxiter = $(maxiter)"
@info "maxtime = $(maxtime)"
@info "targetaccuracy = $(targetaccuracy)"
@info "loggingfreq = $(loggingfreq)"
@info "--------------------------------------------------"


# Running 
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS-sss")
@info "timestamp = $(timestamp)"
@info "Completed initialization."
# loggingfilename = "$(outputdir)/$(dataset)-$(ARGS[2])-$(join(ARGS[3:end], "_"))-execution_log-$(timestamp).txt"
# io = open(loggingfilename, "w+")
# logger = SimpleLogger(io)
outputfilename = "$(outputdir)/$(dataset)-$(elasticnet_λ₁)_$(elasticnet_λ₂)-$(algorithm)-$(parsed_args["lipschitz"])-output-$(timestamp).jld2"
@info "outputfilename = $(outputfilename)"
@info "--------------------------------------------------"


if algorithm == "ACODER"
    @info "Running ACODER..."

    L = parsed_args["lipschitz"]
    γ = parsed_args["gamma"]
    @info "Setting L=$(L), γ=$(γ)"

    acoder_params = ACODERParams(L, γ)
    output_acoder = acoder(problem, exitcriterion, acoder_params)
    save_object(outputfilename, output_acoder)
    @info "output saved to $(outputfilename)"


elseif algorithm == "GD"
    @info "Running GD..."

    L = parsed_args["lipschitz"]
    @info "Setting L=$(L)"

    gd_params = GDParams(L)
    output = gd(problem, exitcriterion, gd_params)
    save_object(outputfilename, output)
    @info "output saved to $(outputfilename)"
# elseif ARGS[2] == "CODERVR"
#     @info "Running CODERVR..."

#     L = parse(Float64, ARGS[3])
#     γ = parse(Float64, ARGS[4])
#     M = parse(Float64, ARGS[5])
#     K = parse(Float64, ARGS[6])
#     @info "Setting L=$(L), γ=$(γ), M=$(M), K=$(K)"

#     codervr_params = CODERVRParams(L, M, γ, K)
#     output_vrcoder = codervr(problem, exitcriterion, codervr_params)
#     save_object(outputfilename, output_vrcoder)
#     @info "output saved to $(outputfilename)"

else
    @info "Wrong algorithm name supplied"
end
