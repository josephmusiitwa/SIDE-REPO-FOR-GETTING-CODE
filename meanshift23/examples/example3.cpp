#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

int main(int argc, char* argv[]) {
    // 1. Specify the input file
    std::string csvFilePath = "data/input/my_data.csv";
    if (argc > 1) {
        csvFilePath = argv[1];
    }
    
    meanshift::Dataset dataset;
    std::cout << "Loading data from external CSV file: " << csvFilePath << std::endl;
    
    // 2. Load the external CSV data
    if (!dataset.loadFromCSV(csvFilePath, true)) {
        std::cerr << "Error: Failed to load external data." << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded external data." << std::endl;
    std::cout << "Running MeanShift clustering..." << std::endl;
    
    // 3. Set up MeanShift algorithm and fit the data
    meanshift::MeanShift ms(2.0, 1e-3, 300);
    ms.fit(dataset.getData());
    
    // 4. Output the results ONLY to the terminal
    auto clusters = ms.getClusters();
    std::cout << "\nNumber of clusters found: " << clusters.size() << std::endl;
    
    int i = 0;
    for (const auto& c : clusters) {
        std::cout << "Cluster " << i++ << " centroid: (";
        for (std::size_t d = 0; d < c.centroid.size(); ++d) {
            std::cout << c.centroid[d] << (d == c.centroid.size() - 1 ? "" : ", ");
        }
        std::cout << ") with " << c.pointIndices.size() << " points." << std::endl;
    }
    
    std::cout << "Clustering evaluation score: " << ms.evaluate(dataset.getData()) << std::endl;
    
    // We intentionally stop here without writing anything to data/output/
    return 0;
}
