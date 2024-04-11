# datanetJSON

This project creates samples for RouteNet-Fermi by parsing JSON files from network simulations and testbeds instead of the original plain text files generated by the RouteNet-Fermi authors' simulator.

The format of the files is described below:

**graph_output.txt:**

The format for the topology information is the same as the format generated by the simulator of the authors of RouteNet-Fermi, described here: https://github.com/BNN-UPC/BNNetSimulator/blob/main/input_parameters_glossary.ipynb

**traffic.json:**
```
{
      "global" : {
            "max_bandwidth": <insert maximum bandwidth of the flows>,
            "duration": <insert duration of the traffic flows>
      },
      "flows": [{
            "src": <insert ID of src node>,
            "dst": <insert ID of dst node>,
            "time_dist": <insert "Poisson",  "CBR", or "On-Off">,
            "pkt_size": <insert average packet size>,
            "avg_bw": <insert average bandwidth in bits per second>,
            "avg_time_on": <insert avg time on in seconds if time_dist is On-Off, otherwise set to -1>,
            "avg_time_off": <insert avg time off in seconds if time_dist is On-Off, otherwise set to -1>
      }, {
            "src": <insert ID of src node>,
            "dst": <insert ID of dst node>,
            "time_dist": <insert "Poisson",  "CBR", or "On-Off">,
            "avg_bw": <insert average bandwidth in bits per second>,
            "avg_time_on": <insert avg time on in seconds if time_dist is On-Off, otherwise set to -1>,
            "avg_time_off": <insert avg time off in seconds if time_dist is On-Off, otherwise set to -1>
      },
           ....
      }]
}
```

**simulationResults.json:**
```
{
      "global" : {
            "avg_packets_per_second": <insert total average packets per second in the whole network>,
            "avg_packets_lost_per_second": <insert total average packets lost per second in the whole network>,
            "avg_packet_delay": <insert the global average packet delay in the network, this includes all flows>
      },
      "flows": [{
            "src": <insert ID of src node>,
            "dst": <insert ID of dst node>,
            "avg_bw": <insert average bandwidth in bits per second>,
            "total_packets_transmitted": <insert the total number of packets sent>,
            "total_packets_lost": <insert the total number of packets lost>,
            "avg_delay": <insert avg packet delay>,
            "delay_variance": <insert variance of delay>,
            "avg_ln_delay": <insert avg of ln(packet_delay) (-1 if not available)>,
            "10_percentile_delay": <insert 10th percentile of avg delay (-1 if not available)>,
            "20_percentile_delay": <insert 20th percentile of avg delay (-1 if not available)>,
            "50_percentile_delay": <insert 50th percentile of avg delay (-1 if not available)>,
            "80_percentile_delay": <insert 80th percentile of avg delay (-1 if not available)>,
            "90_percentile_delay": <insert 90th percentile of avg delay (-1 if not available)>,
      }, {
            "src": <insert ID of src node>,
            "dst": <insert ID of dst node>,
            "avg_bw": <insert average bandwidth in bits per second>,
            "total_packets_transmitted": <insert the total number of packets sent>,
            "total_packets_lost": <insert the total number of packets lost>,
            "avg_delay": <insert avg packet delay>,
            "delay_variance": <insert variance of delay>,
            "avg_ln_delay": <insert avg of ln(packet_delay) (-1 if not available)>,
            "10_percentile_delay": <insert 10th percentile of avg delay (-1 if not available)>,
            "20_percentile_delay": <insert 20th percentile of avg delay (-1 if not available)>,
            "50_percentile_delay": <insert 50th percentile of avg delay (-1 if not available)>,
            "80_percentile_delay": <insert 80th percentile of avg delay (-1 if not available)>,
            "90_percentile_delay": <insert 90th percentile of avg delay (-1 if not available)>,
      }
           ....
      }]
}
```

**linkUsage.json:**
```
{
      "links": [{
            "src": <insert ID of src node>,
            "dst": <insert ID of dst node>,
            "port": <insert port number from the src node>,
            "avg_utilization": <insert avg utilization of the port in range [0,1]>,
            "avg_packets_lost": <insert avg packets lost in the port in range [0,1]>,
            "avg_packet_size": <insert the avg packet size going through the port>
      }, {
            "src": <insert ID of src node>,
            "dst": <insert ID of dst node>,
            "port": <insert port number from the src node that takes us to the dst node>,
            "avg_utilization": <insert avg utilization of the port in range [0,1]>,
            "avg_packets_lost": <insert avg packets lost in the port in range [0,1]>,
            "avg_packet_size": <insert the avg packet size going through the port>
      }
           ....
      }]
}
```
