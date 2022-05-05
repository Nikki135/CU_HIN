import argparse
import pandas as pd
import scipy.sparse as sp
from dataprun import GenerateWL
import logging
from domain2IP_matrix import *
import csv
#import maliciousdetect

def createCSR(df):
    '''
    Creates CSR matrix from Pandas dataframe object. Pandas dataframe must contain a
    'src' and 'dest' representing integer values of IPs from a dictionary. 
    Note: no NaN's allowed.
    '''
    print("inside createCSR function")
#    print(df)
#    print("domaintoIP", domain2ip)
    # Find and count number of occurrences of repeated IP pairs 
    pairindex = df.groupby(['src', 'dest']).indices
    paircount = {k: len(v) for k, v in pairindex.items()}

    # Extracting src, dest, counts
    xypair = list(paircount.keys())
    cols = [i[0] for i in xypair]                 # Setting src/'x' to be column
    rows = [i[1] for i in xypair]                 # Setting dest/'y' to be row
    vals = list(paircount.values())               # Values

    # Create Compressed Sparse Row Matrix
    ip2ipmatrix = sp.csr_matrix((vals, (cols, rows)))

    return ip2ipmatrix

#def search_domain(
def ip_to_ip(ip2index: dict, filenames: list, domain2ip: dict):
    '''
    Ip_to_ip.py creates the ip to ip csr matrix for hindom project. 

    Arguments:
    ip2index: dict - Mapping from 
    filenames: list - The files with the netflow.

    Example: 
    python ip_to_ip.py --dns_files /data/dns/2021-04-10_dns.00:00:00-01:00:00.log 
                       --netflow_files /data/converted/ft-v05.2021-04-10.000000-0600.csv
    '''
#    print("Domain to IP:", domain2ip)
    print("inside ip_to_ip function")
    # Extract SRC and DEST IPs addresses as though from a csv file and 
    # create a Pandas dataframe
    filename = filenames[0]
    ip2ip =  pd.read_csv(filename, sep=',', header=0, usecols=[10, 11], 
                            names=['src', 'dest'], engine='python')
#    print("ip2ip",ip2ip['dest'])
#    print("typeofip", type(ip2ip))
    for i in range(1, len(filenames)):
        filename = filenames[i]
        with open(filename, 'r') as infile:
            more = pd.read_csv(filename, sep='\\t', header=0, usecols=[10, 11], 
                            names=['src', 'dest'], engine='python')
            ip2ip = ip2ip.concat(more)
            
            
#============================= CODE TO GET DESTIP2DOMAIN MATRIX =================================================
    destipaddr = ip2ip['dest'].values.tolist()


    destip2domain = {}

    l1 = []
#    for i in range(0, len(destipaddr)):
    ip_to_domain = []
    for search_val in destipaddr:
        for key, val in domain2ip.items():
#            print("val", val)
            if search_val in val:
#                print("search value", search_val)
#                print("inside if", val)
#                print("")
#                destip2domain.update({search_val:key})
#                for r in range(0, len):
#                for key in destip2domain:
                ip_to_domain.append([val, key])
        print(ip_to_domain)
#        DF = pd.DataFrame(ip_to_domain)
#        DF.to_csv("destdomaindata.csv")


#                with open('destip2domain.csv', 'a', newline='') as csvfile:
#                    fieldnames = ['ip_addr', 'domain_name']
#                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                    for key in destip2domain:
#                        writer.writerow({'ip_addr': key, 'domain_name': destip2domain[key]})
##                l1.append(key)
##                print("IP to domain matrix", destip2domain)
#                break
            

#    print("IP to domain matrix", destip2domain)
#    with open('destip2domain2.csv', 'w', newline='') as csvfile:
#        fieldnames = ['ip_addr', 'domain_name']
#        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#        writer.writeheader()
#        for key in destip2domain:
#            writer.writerow({'ip_addr': key, 'domain_name': destip2domain[key]})

        
        
    indices = []
    for index, row in ip2ip.iterrows():
      if row['src'] not in ip2index or row['dest'] not in ip2index:
        indices.append(index)
   
    lenbefore = len(ip2ip)
    ip2ip = ip2ip.drop(indices) 
    logging.info("Kept " + str(float(100* len(ip2ip))/lenbefore)  + 
                 "% of netflow rows.")
    
    # Convert to integers
    ip2ip['src'] = ip2ip['src'].map(ip2index)
#    print("src", ip2ip['src'])         # Map IP's to index values
    ip2ip['dest'] = ip2ip['dest'].map(ip2index)
#    print("dst", ip2ip['dest'])     # " " "
    ip2ip = ip2ip.astype({'src': int, 'dest': int})   # Convert to integers      
#    print("IP2IPmatrixis:", ip2ip)
    # Create CSR 
    ip2ipmatrix = createCSR(ip2ip)
#    print("ip2ipmatrix", ip2ipmatrix)
    return ip2ipmatrix


if __name__ == '__main__':
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dns_files", type=str, nargs='+', required=True,
      help="The dns log file(s) to use.")
    parser.add_argument('--netflow_files', type=str, required=True, nargs='+',
                        help='Expects log file from /data/dns directory')

    FLAGS = parser.parse_args()
#      RL, domain2index, ip2index =  dataprun.GenerateWL(FLAGS.dns_files)
  #global domain2ip
#  domain2ip = dataprun.GenerateDomain2IP(RL, domain2index)
    RL, domain2index, ip2index =  dataprun.GenerateWL(FLAGS.dns_files)
    domain2ip = dataprun.GenerateDomain2IP(RL, domain2index)
    ip_to_ip(ip2index, FLAGS.netflow_files, domain2ip)
