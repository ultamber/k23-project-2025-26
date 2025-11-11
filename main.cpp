#include <iostream>
#include <fstream>
#include <memory>
#include "include/arguments.hpp"
#include "include/dataset.hpp"
#include "include/algorithms/lsh.hpp"
#include "include/algorithms/hypercube.hpp"
#include "include/algorithms/clustering/ivfflat.hpp"
#include "include/algorithms/clustering/ivfpq.hpp"
#include "include/search_method.hpp"

Arguments parseArgs(int argc, char *argv[])
{
    Arguments a;
    for (int i = 1; i < argc; ++i)
    {
        std::string flag = argv[i];
        if (flag == "-d" && i + 1 < argc)
            a.inputFile = argv[++i];
        else if (flag == "-q" && i + 1 < argc)
            a.queryFile = argv[++i];
        else if (flag == "-o" && i + 1 < argc)
            a.outputFile = argv[++i];
        else if (flag == "-gt" && i + 1 < argc)
            a.gtFile = argv[++i];
        else if (flag == "-type" && i + 1 < argc)
            a.type = argv[++i];
        else if (flag == "-lsh")
            a.useLSH = true;
        else if (flag == "-hypercube")
            a.useHypercube = true;
        else if (flag == "-ivfflat")
            a.useIVFFlat = true;
        else if (flag == "-ivfpq")
            a.useIVFPQ = true;
        else if (flag == "-bruteforce")
            a.useBruteForce = true;
        else if (flag == "-N" && i + 1 < argc)
            a.N = std::stoi(argv[++i]);
        else if (flag == "-R" && i + 1 < argc)
            a.R = std::stod(argv[++i]);
        else if (flag == "-seed" && i + 1 < argc)
            a.seed = std::stoi(argv[++i]);
        else if (flag == "-range" && i + 1 < argc)
        {
            std::string v = argv[++i];
            a.rangeSearch = (v == "true" || v == "1");
        }
        else if (flag == "-k" && i + 1 < argc)
            a.k = std::stoi(argv[++i]);
        else if (flag == "-L" && i + 1 < argc)
            a.L = std::stoi(argv[++i]);
        else if (flag == "-w" && i + 1 < argc)
            a.w = std::stod(argv[++i]);
        else if (flag == "-kproj" && i + 1 < argc)
            a.kproj = std::stoi(argv[++i]);
        else if (flag == "-M" && i + 1 < argc)
            a.M = std::stoi(argv[++i]);
        else if (flag == "-probes" && i + 1 < argc)
            a.probes = std::stoi(argv[++i]);
        else if (flag == "-kclusters" && i + 1 < argc)
            a.kclusters = std::stoi(argv[++i]);
        else if (flag == "-nprobe" && i + 1 < argc)
            a.nprobe = std::stoi(argv[++i]);
        else if (flag == "-nbits" && i + 1 < argc)
            a.nbits = std::stoi(argv[++i]);
        else if (flag == "-Msub" && i + 1 < argc)
            a.Msubvectors = std::stoi(argv[++i]);
    }
    return a;
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: ./search -d <input> -q <query> -o <output> -type mnist -lsh|-hypercube|-ivfflat|-ivfpq [params]\n";
        return 1;
    }

    Arguments args = parseArgs(argc, argv);
    std::cout << "Input: " << args.inputFile << "\nQuery: " << args.queryFile
              << "\nOutput: " << args.outputFile << "\n";

    std::unique_ptr<Dataset> data, queries;
    if (args.type == "mnist"){
        data = std::make_unique<MNIST_Dataset>();
        queries = std::make_unique<MNIST_Dataset>();
    }
    else if (args.type == "sift"){
        data = std::make_unique<SIFT_Dataset>();
        queries = std::make_unique<SIFT_Dataset>();
    }
    else{
        std::cerr << "Error: Unknown type. Valid values: \"mnist\" and \"sift\". \n";
        return 1;
    }
    data->load(args.inputFile);
    queries->load(args.queryFile);

    std::ofstream out(args.outputFile);
    if (!out)
    {
        std::cerr << "Cannot open output file.\n";
        return 1;
    }

    std::unique_ptr<SearchMethod> alg;

    if (args.useLSH)
    {
        alg = std::make_unique<LSH>(args, data->dimension, data->vectors);
    }
    else if (args.useHypercube)
    {
        alg = std::make_unique<Hypercube>(args, data->dimension, data->vectors);
    }
    else if (args.useIVFFlat)
    {
        alg = std::make_unique<IVFFlat>(args, data->dimension, data->vectors);
    }
    else if (args.useIVFPQ)
    {
        alg = std::make_unique<IVFPQ>(args, data->dimension, data->vectors);
    }
    else
    {
        std::cerr << "Error: specify -lsh or -hypercube or -ivfflat or -ivfpq\n";
        return 1;
    }

    alg->buildIndex();
    alg->setUpGroundTruth(queries->vectors);
    alg->search(queries->vectors, out);

    std::cout << "Search completed.\n";
    return 0;
}
