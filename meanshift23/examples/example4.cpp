#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
int main() {
    std::string csvFilePath;
    std::string outFilePath;
    
    std::cout << "--- MeanShift Clustering Interactive Tool ---" << std::endl;
    std::cout << "Please enter the path to your input CSV file (e.g., data/input/my_data.csv): ";
    std::cin >> csvFilePath;
    
    std::cout << "Please enter the path where you want to save the results (e.g., data/output/results.csv): ";
    std::cin >> outFilePath;
    
    meanshift::Dataset dataset;
    std::cout << "\nLoading data from external CSV file: " << csvFilePath << std::endl;
    
    if (!dataset.loadFromCSV(csvFilePath, true)) {
        std::cerr << "Error: Failed to load external data. Please check the spelling of your file path." << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded external data." << std::endl;
    std::cout << "Running MeanShift clustering..." << std::endl;
    
    meanshift::MeanShift ms(2.0, 1e-3, 300);
    ms.fit(dataset.getData());
    
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
    
    std::cout << "\nWriting results to: " << outFilePath << " ..." << std::endl;
    
    std::ofstream outFile(outFilePath);
    if (outFile.is_open()) {
        outFile << "Feature1,Feature2,Cluster_ID\n";
        
        const auto& realData = dataset.getData();
        const auto& labels = ms.getLabels();
        
        for (std::size_t k = 0; k < realData.size(); ++k) {
            for (std::size_t d = 0; d < realData[k].size(); ++d) {
                outFile << realData[k][d] << ",";
            }
            outFile << labels[k] << "\n";
        }
        outFile.close();
        std::cout << "Successfully saved!" << std::endl;
    } else {
        std::cerr << "Warning: Could not open output file for writing! Check if the output directory exists." << std::endl;
    }

    return 0;
}
