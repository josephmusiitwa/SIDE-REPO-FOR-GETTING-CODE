#include "meanshift/Dataset.hpp"
#include "meanshift/MeanShift.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

int main() {
    meanshift::Dataset dataset;
    
    // Create some dummy data for the example if no CSV is provided
    std::cout << "Creating dummy dataset for example..." << std::endl;
    // Assuming we could load: dataset.loadFromCSV("data/input/data.csv", true);
    
    std::cout << "Running MeanShift clustering..." << std::endl;
    meanshift::MeanShift ms(2.0, 1e-3, 300);
    
    // Dummy manual data
    // In practice use dataset.getData()
    meanshift::Data data = {
        {1.0, 1.0}, {1.5, 2.0}, {2.0, 1.0},
        {8.0, 8.0}, {8.5, 9.0}, {9.0, 8.0},
        {1.0, 8.0}, {1.5, 9.0}, {2.0, 8.0}
    };
    
    ms.fit(data);
    
    auto clusters = ms.getClusters();
    std::cout << "Number of clusters found: " << clusters.size() << std::endl;
    
    int i = 0;
    for (const auto& c : clusters) {
        std::cout << "Cluster " << i++ << " centroid: (";
        for (std::size_t d = 0; d < c.centroid.size(); ++d) {
            std::cout << c.centroid[d] << (d == c.centroid.size() - 1 ? "" : ", ");
        }
        std::cout << ") with " << c.pointIndices.size() << " points." << std::endl;
    }
    
    std::cout << "Clustering evaluation score (cohesion): " << ms.evaluate(data) << std::endl;
    
    return 0;
}
