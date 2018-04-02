from concurrent.futures import *
from time import *
import multiprocessing

threadCount = multiprocessing.cpu_count()

print( " + ThreadCount: " + str( threadCount ) )
executor = ThreadPoolExecutor( threadCount )
futures = set()

def work( i ):
    print( " - " + str( i ) + " begin" )
    sleep( 0.5 )
    print( " - " + str( i ) + " end" )

for i in range( 50 ):
    sleep( 0.01 )
    futures.add( executor.submit( work, i ) )

print( " + Mainthread waits" )
wait( futures )
print( " + Mainthread finished" )
