	  �     k820309              12.1        �3]U                                                                                                           
       conjugate_gradient_method.f90 CONJUGATE_GRADIENT                                                    
                                                                                                           8                  @               A                '�                   #X    #L    #RESIDUE    #PDIRECTION               �                                                         
            &                   &                                                                   ��*            y
                                                          �                                        `                
            &                   &                                                                   ��*            y
                                                          �                                          �                 
            &                   &                                                     �                                                           
            &                   &                                                                                           NULL #         @                                  	                 #CG_INITIALIZE%NGLOB_AB 
   #CG_INITIALIZE%SIZE    #CG_VARIABLE    #DISPL    #LOAD                                                
                                                            SIZE           D                                      �              #CG_DATA             
                                                     
    p          p          5 r 
       p          5 r 
                              
                                                     
    p          p          5 r 
       p          5 r 
                     #         @                                                   #UPDATE_VALUE_DIRECTION%NGLOB_AB    #UPDATE_VALUE_DIRECTION%ABS    #UPDATE_VALUE_DIRECTION%MINVAL    #UPDATE_VALUE_DIRECTION%MAXVAL    #UPDATE_VALUE_DIRECTION%SIZE    #CG_VARIABLE                                                                                                            ABS                                                 MINVAL                                                 MAXVAL                                                 SIZE           
D                                      �              #CG_DATA    #         @                                                   #AXX%NGLOB_AB    #AX    #X                                                                     D @                                                  
 	    p          p          5 r        p          5 r                               
@ @                                                  
    p          p          5 r        p          5 r                      #         @                                                   #VECTOR_MULTIPLICATION%NGLOB_AB    #SUM_ALL    #A    #B                                                                      D @                                   
                
                                                     
 
   p          p          5 r        p          5 r                               
                                                     
    p          p          5 r        p          5 r                         �   9      fn#fn    �   @   J   CONSTANTS &     q       CUSTOM_REAL+CONSTANTS    �  {       CG_DATA        a   CG_DATA%X        a   CG_DATA%L       �   a   CG_DATA%RESIDUE #   �  �   a   CG_DATA%PDIRECTION    u  =       NULL    �  �       CG_INITIALIZE 3   T  @     CG_INITIALIZE%NGLOB_AB+SPECFEM_PAR #   �  =      CG_INITIALIZE%SIZE *   �  U   a   CG_INITIALIZE%CG_VARIABLE $   &  �   a   CG_INITIALIZE%DISPL #   �  �   a   CG_INITIALIZE%LOAD '   �        UPDATE_VALUE_DIRECTION <   �	  @     UPDATE_VALUE_DIRECTION%NGLOB_AB+SPECFEM_PAR +   �	  <      UPDATE_VALUE_DIRECTION%ABS .   
  ?      UPDATE_VALUE_DIRECTION%MINVAL .   N
  ?      UPDATE_VALUE_DIRECTION%MAXVAL ,   �
  =      UPDATE_VALUE_DIRECTION%SIZE 3   �
  U   a   UPDATE_VALUE_DIRECTION%CG_VARIABLE      i       AXX )   �  @     AXX%NGLOB_AB+SPECFEM_PAR    �  �   a   AXX%AX    |  �   a   AXX%X &   0  �       VECTOR_MULTIPLICATION ;   �  @     VECTOR_MULTIPLICATION%NGLOB_AB+SPECFEM_PAR .   �  @   a   VECTOR_MULTIPLICATION%SUM_ALL (   7  �   a   VECTOR_MULTIPLICATION%A (   �  �   a   VECTOR_MULTIPLICATION%B 