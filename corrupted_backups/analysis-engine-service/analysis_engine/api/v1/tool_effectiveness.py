"""
Tool effectiveness module.

This module provides functionality for...
"""

f r o m   f a s t a p i   i m p o r t   A P I R o u t e r ,   D e p e n d s ,   H T T P E x c e p t i o n ,   Q u e r y 
 
 f r o m   t y p i n g   i m p o r t   D i c t ,   L i s t ,   O p t i o n a l 
 
 i m p o r t   d a t e t i m e 
 
 f r o m   e n u m   i m p o r t   E n u m 
 
 f r o m   p y d a n t i c   i m p o r t   B a s e M o d e l ,   F i e l d 
 
 f r o m   s q l a l c h e m y . o r m   i m p o r t   S e s s i o n 
 
 
 
 #   I m p o r t   t h e   T o o l   E f f e c t i v e n e s s   s e r v i c e   i m p l e m e n t a t i o n 
 
 f r o m   a n a l y s i s _ e n g i n e . s e r v i c e s . t o o l _ e f f e c t i v e n e s s   i m p o r t   ( 
 
         T o o l E f f e c t i v e n e s s T r a c k e r ,   
 
         S i g n a l ,   
 
         S i g n a l O u t c o m e , 
 
         W i n R a t e M e t r i c ,   
 
         P r o f i t F a c t o r M e t r i c ,   
 
         E x p e c t e d P a y o f f M e t r i c , 
 
         R e l i a b i l i t y B y M a r k e t R e g i m e M e t r i c 
 
 ) 
 
 
 
 #   I m p o r t   d a t a b a s e   d e p e n d e n c i e s 
 
 f r o m   a n a l y s i s _ e n g i n e . d b . c o n n e c t i o n   i m p o r t   g e t _ d b _ s e s s i o n 
 
 f r o m   a n a l y s i s _ e n g i n e . r e p o s i t o r i e s . t o o l _ e f f e c t i v e n e s s _ r e p o s i t o r y   i m p o r t   T o o l E f f e c t i v e n e s s R e p o s i t o r y 
 
 
 
 r o u t e r   =   A P I R o u t e r ( ) 
 
 
 
 #   D e f i n e   A P I   m o d e l s 
 
 c l a s s   M a r k e t R e g i m e E n u m ( s t r ,   E n u m ) : 
 
         T R E N D I N G   =   " t r e n d i n g " 
 
         R A N G I N G   =   " r a n g i n g " 
 
         V O L A T I L E   =   " v o l a t i l e " 
 
         U N K N O W N   =   " u n k n o w n " 
 
 
 
 c l a s s   S i g n a l R e q u e s t ( B a s e M o d e l ) : 
 
         t o o l _ i d :   s t r   =   F i e l d ( . . . ,   d e s c r i p t i o n = " I d e n t i f i e r   f o r   t h e   t r a d i n g   t o o l   t h a t   g e n e r a t e d   t h e   s i g n a l " ) 
 
         s i g n a l _ t y p e :   s t r   =   F i e l d ( . . . ,   d e s c r i p t i o n = " T y p e   o f   s i g n a l   ( b u y ,   s e l l ,   e t c . ) " ) 
 
         i n s t r u m e n t :   s t r   =   F i e l d ( . . . ,   d e s c r i p t i o n = " T r a d i n g   i n s t r u m e n t   ( e . g . ,   ' E U R _ U S D ' ) " ) 
 
         t i m e s t a m p :   d a t e t i m e . d a t e t i m e   =   F i e l d ( . . . ,   d e s c r i p t i o n = " W h e n   t h e   s i g n a l   w a s   g e n e r a t e d " ) 
 
         c o n f i d e n c e :   f l o a t   =   F i e l d ( d e f a u l t = 1 . 0 ,   g e = 0 . 0 ,   l e = 1 . 0 ,   d e s c r i p t i o n = " S i g n a l   c o n f i d e n c e   l e v e l   ( 0 . 0 - 1 . 0 ) " ) 
 
         t i m e f r a m e :   s t r   =   F i e l d ( . . . ,   d e s c r i p t i o n = " T i m e f r a m e   o f   t h e   a n a l y s i s   ( e . g . ,   ' 1 H ' ,   ' 4 H ' ,   ' 1 D ' ) " ) 
 
         m a r k e t _ r e g i m e :   M a r k e t R e g i m e E n u m   =   F i e l d ( d e f a u l t = M a r k e t R e g i m e E n u m . U N K N O W N ,   d e s c r i p t i o n = " M a r k e t   r e g i m e   a t   s i g n a l   t i m e " ) 
 
         a d d i t i o n a l _ d a t a :   O p t i o n a l [ D i c t ]   =   F i e l d ( d e f a u l t = N o n e ,   d e s c r i p t i o n = " A n y   a d d i t i o n a l   s i g n a l   m e t a d a t a " ) 
 
 
 
 c l a s s   O u t c o m e R e q u e s t ( B a s e M o d e l ) : 
 
         s i g n a l _ i d :   s t r   =   F i e l d ( . . . ,   d e s c r i p t i o n = " I D   o f   t h e   s i g n a l   t h a t   t h i s   o u t c o m e   i s   a s s o c i a t e d   w i t h " ) 
 
         s u c c e s s :   b o o l   =   F i e l d ( . . . ,   d e s c r i p t i o n = " W h e t h e r   t h e   s i g n a l   l e d   t o   a   s u c c e s s f u l   t r a d e " ) 
 
         r e a l i z e d _ p r o f i t :   f l o a t   =   F i e l d ( d e f a u l t = 0 . 0 ,   d e s c r i p t i o n = " P r o f i t / l o s s   r e a l i z e d   f r o m   t h e   t r a d e " ) 
 
         t i m e s t a m p :   d a t e t i m e . d a t e t i m e   =   F i e l d ( . . . ,   d e s c r i p t i o n = " W h e n   t h e   o u t c o m e   w a s   r e c o r d e d " ) 
 
         a d d i t i o n a l _ d a t a :   O p t i o n a l [ D i c t ]   =   F i e l d ( d e f a u l t = N o n e ,   d e s c r i p t i o n = " A n y   a d d i t i o n a l   o u t c o m e   m e t a d a t a " ) 
 
 
 
 c l a s s   E f f e c t i v e n e s s M e t r i c ( B a s e M o d e l ) : 
 
         n a m e :   s t r 
 
         v a l u e :   f l o a t 
 
         d e s c r i p t i o n :   s t r 
 
 
 
 c l a s s   T o o l E f f e c t i v e n e s s ( B a s e M o d e l ) : 
 
         t o o l _ i d :   s t r 
 
         m e t r i c s :   L i s t [ E f f e c t i v e n e s s M e t r i c ] 
 
         s i g n a l _ c o u n t :   i n t 
 
         f i r s t _ s i g n a l _ d a t e :   O p t i o n a l [ d a t e t i m e . d a t e t i m e ] 
 
         l a s t _ s i g n a l _ d a t e :   O p t i o n a l [ d a t e t i m e . d a t e t i m e ] 
 
         s u c c e s s _ r a t e :   f l o a t 
 
 
 
 #   S i n g l e t o n   i n s t a n c e   o f   t h e   e f f e c t i v e n e s s   t r a c k e r 
 
 t r a c k e r   =   T o o l E f f e c t i v e n e s s T r a c k e r ( ) 
 
 
 
 @ r o u t e r . p o s t ( " / s i g n a l s / " ,   s t a t u s _ c o d e = 2 0 1 ,   r e s p o n s e _ m o d e l = D i c t [ s t r ,   s t r ] ) 
 
 d e f   r e g i s t e r _ s i g n a l ( s i g n a l _ d a t a :   S i g n a l R e q u e s t ,   d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) ) : 
 
         " " " 
 
         R e g i s t e r   a   n e w   s i g n a l   f r o m   a   t r a d i n g   t o o l   f o r   e f f e c t i v e n e s s   t r a c k i n g 
 
         " " " 
 
         #   C r e a t e   S i g n a l   o b j e c t   f o r   i n - m e m o r y   t r a c k i n g 
 
         s i g n a l   =   S i g n a l ( 
 
                 t o o l _ i d = s i g n a l _ d a t a . t o o l _ i d , 
 
                 s i g n a l _ t y p e = s i g n a l _ d a t a . s i g n a l _ t y p e , 
 
                 i n s t r u m e n t = s i g n a l _ d a t a . i n s t r u m e n t , 
 
                 t i m e s t a m p = s i g n a l _ d a t a . t i m e s t a m p , 
 
                 c o n f i d e n c e = s i g n a l _ d a t a . c o n f i d e n c e , 
 
                 t i m e f r a m e = s i g n a l _ d a t a . t i m e f r a m e , 
 
                 m a r k e t _ r e g i m e = s i g n a l _ d a t a . m a r k e t _ r e g i m e . v a l u e , 
 
                 a d d i t i o n a l _ d a t a = s i g n a l _ d a t a . a d d i t i o n a l _ d a t a   o r   { } 
 
         ) 
 
         
 
         #   R e g i s t e r   i n   m e m o r y 
 
         s i g n a l _ i d   =   t r a c k e r . r e g i s t e r _ s i g n a l ( s i g n a l ) 
 
         
 
         #   S a v e   t o   d a t a b a s e 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         
 
         #   E n s u r e   t h e   t o o l   e x i s t s 
 
         t o o l   =   r e p o s i t o r y . g e t _ t o o l ( s i g n a l _ d a t a . t o o l _ i d ) 
 
         i f   n o t   t o o l : 
 
                 #   C r e a t e   t h e   t o o l   i f   i t   d o e s n ' t   e x i s t 
 
                 r e p o s i t o r y . c r e a t e _ t o o l ( { 
 
                         " t o o l _ i d " :   s i g n a l _ d a t a . t o o l _ i d , 
 
                         " n a m e " :   s i g n a l _ d a t a . t o o l _ i d ,     #   U s e   t o o l _ i d   a s   n a m e   b y   d e f a u l t 
 
                 } ) 
 
         
 
         #   S a v e   s i g n a l   t o   d a t a b a s e 
 
         r e p o s i t o r y . c r e a t e _ s i g n a l ( { 
 
                 " s i g n a l _ i d " :   s i g n a l _ i d , 
 
                 " t o o l _ i d " :   s i g n a l _ d a t a . t o o l _ i d , 
 
                 " s i g n a l _ t y p e " :   s i g n a l _ d a t a . s i g n a l _ t y p e , 
 
                 " i n s t r u m e n t " :   s i g n a l _ d a t a . i n s t r u m e n t , 
 
                 " t i m e s t a m p " :   s i g n a l _ d a t a . t i m e s t a m p , 
 
                 " c o n f i d e n c e " :   s i g n a l _ d a t a . c o n f i d e n c e , 
 
                 " t i m e f r a m e " :   s i g n a l _ d a t a . t i m e f r a m e , 
 
                 " m a r k e t _ r e g i m e " :   s i g n a l _ d a t a . m a r k e t _ r e g i m e . v a l u e , 
 
                 " a d d i t i o n a l _ d a t a " :   s i g n a l _ d a t a . a d d i t i o n a l _ d a t a   o r   { } 
 
         } ) 
 
         
 
         r e t u r n   { " s i g n a l _ i d " :   s i g n a l _ i d } 
 
 
 
 @ r o u t e r . p o s t ( " / o u t c o m e s / " ,   s t a t u s _ c o d e = 2 0 1 ) 
 
 d e f   r e g i s t e r _ o u t c o m e ( o u t c o m e _ d a t a :   O u t c o m e R e q u e s t ,   d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) ) : 
 
         " " " 
 
         R e g i s t e r   t h e   o u t c o m e   o f   a   p r e v i o u s l y   r e g i s t e r e d   s i g n a l 
 
         " " " 
 
         #   C r e a t e   o u t c o m e   o b j e c t   f o r   i n - m e m o r y   t r a c k i n g 
 
         o u t c o m e   =   S i g n a l O u t c o m e ( 
 
                 s i g n a l _ i d = o u t c o m e _ d a t a . s i g n a l _ i d , 
 
                 s u c c e s s = o u t c o m e _ d a t a . s u c c e s s , 
 
                 r e a l i z e d _ p r o f i t = o u t c o m e _ d a t a . r e a l i z e d _ p r o f i t , 
 
                 t i m e s t a m p = o u t c o m e _ d a t a . t i m e s t a m p , 
 
                 a d d i t i o n a l _ d a t a = o u t c o m e _ d a t a . a d d i t i o n a l _ d a t a   o r   { } 
 
         ) 
 
         
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         
 
         #   C h e c k   i f   s i g n a l   e x i s t s   i n   d a t a b a s e 
 
         s i g n a l   =   r e p o s i t o r y . g e t _ s i g n a l ( o u t c o m e _ d a t a . s i g n a l _ i d ) 
 
         i f   n o t   s i g n a l : 
 
                 r a i s e   H T T P E x c e p t i o n ( s t a t u s _ c o d e = 4 0 4 ,   d e t a i l = f " S i g n a l   n o t   f o u n d :   { o u t c o m e _ d a t a . s i g n a l _ i d } " ) 
 
         
 
         t r y : 
 
                 #   R e g i s t e r   i n   m e m o r y 
 
                 t r a c k e r . r e g i s t e r _ o u t c o m e ( o u t c o m e ) 
 
                 
 
                 #   S a v e   t o   d a t a b a s e 
 
                 r e p o s i t o r y . c r e a t e _ o u t c o m e ( { 
 
                         " s i g n a l _ i d " :   o u t c o m e _ d a t a . s i g n a l _ i d , 
 
                         " s u c c e s s " :   o u t c o m e _ d a t a . s u c c e s s , 
 
                         " r e a l i z e d _ p r o f i t " :   o u t c o m e _ d a t a . r e a l i z e d _ p r o f i t , 
 
                         " t i m e s t a m p " :   o u t c o m e _ d a t a . t i m e s t a m p , 
 
                         " a d d i t i o n a l _ d a t a " :   o u t c o m e _ d a t a . a d d i t i o n a l _ d a t a   o r   { } 
 
                 } ) 
 
                 
 
                 #   C a l c u l a t e   a n d   s t o r e   m e t r i c s 
 
                 t o o l _ i d   =   s i g n a l . t o o l _ i d 
 
                 n o w   =   d a t e t i m e . d a t e t i m e . u t c n o w ( ) 
 
                 
 
                 #   U s e   a   l o o k b a c k   p e r i o d   o f   3 0   d a y s   b y   d e f a u l t 
 
                 s t a r t _ d a t e   =   n o w   -   d a t e t i m e . t i m e d e l t a ( d a y s = 3 0 ) 
 
                 
 
                 #   G e t   s i g n a l s   a n d   o u t c o m e s   f o r   t h i s   t o o l 
 
                 s i g n a l s   =   r e p o s i t o r y . g e t _ s i g n a l s ( 
 
                         t o o l _ i d = t o o l _ i d , 
 
                         f r o m _ d a t e = s t a r t _ d a t e , 
 
                         t o _ d a t e = n o w , 
 
                         l i m i t = 1 0 0 0     #   R e a s o n a b l e   l i m i t   f o r   m e t r i c   c a l c u l a t i o n 
 
                 ) 
 
                 
 
                 s i g n a l _ i d s   =   [ s . s i g n a l _ i d   f o r   s   i n   s i g n a l s ] 
 
                 o u t c o m e s   =   [ ] 
 
                 f o r   s i g _ i d   i n   s i g n a l _ i d s : 
 
                         o u t c o m e s . e x t e n d ( r e p o s i t o r y . g e t _ o u t c o m e s _ f o r _ s i g n a l ( s i g _ i d ) ) 
 
                 
 
                 #   C a l c u l a t e   m e t r i c s 
 
                 i f   s i g n a l s   a n d   o u t c o m e s : 
 
                         #   C o n v e r t   D B   m o d e l s   t o   s e r v i c e   m o d e l s   f o r   c a l c u l a t i o n 
 
                         s e r v i c e _ s i g n a l s   =   [ 
 
                                 S i g n a l ( 
 
                                         i d = s t r ( s . s i g n a l _ i d ) , 
 
                                         t o o l _ i d = s . t o o l _ i d , 
 
                                         s i g n a l _ t y p e = s . s i g n a l _ t y p e , 
 
                                         i n s t r u m e n t = s . i n s t r u m e n t , 
 
                                         t i m e s t a m p = s . t i m e s t a m p , 
 
                                         c o n f i d e n c e = s . c o n f i d e n c e , 
 
                                         t i m e f r a m e = s . t i m e f r a m e , 
 
                                         m a r k e t _ r e g i m e = s . m a r k e t _ r e g i m e , 
 
                                         a d d i t i o n a l _ d a t a = s . a d d i t i o n a l _ d a t a 
 
                                 )   f o r   s   i n   s i g n a l s 
 
                         ] 
 
                         
 
                         s e r v i c e _ o u t c o m e s   =   [ 
 
                                 S i g n a l O u t c o m e ( 
 
                                         s i g n a l _ i d = s t r ( o . s i g n a l _ i d ) , 
 
                                         s u c c e s s = o . s u c c e s s , 
 
                                         r e a l i z e d _ p r o f i t = o . r e a l i z e d _ p r o f i t , 
 
                                         t i m e s t a m p = o . t i m e s t a m p , 
 
                                         a d d i t i o n a l _ d a t a = o . a d d i t i o n a l _ d a t a 
 
                                 )   f o r   o   i n   o u t c o m e s 
 
                         ] 
 
                         
 
                         #   C a l c u l a t e   m e t r i c s 
 
                         w i n _ r a t e   =   W i n R a t e M e t r i c ( ) . c a l c u l a t e ( s e r v i c e _ s i g n a l s ,   s e r v i c e _ o u t c o m e s ) 
 
                         p r o f i t _ f a c t o r   =   P r o f i t F a c t o r M e t r i c ( ) . c a l c u l a t e ( s e r v i c e _ s i g n a l s ,   s e r v i c e _ o u t c o m e s ) 
 
                         e x p e c t e d _ p a y o f f   =   E x p e c t e d P a y o f f M e t r i c ( ) . c a l c u l a t e ( s e r v i c e _ s i g n a l s ,   s e r v i c e _ o u t c o m e s ) 
 
                         
 
                         #   S t o r e   m e t r i c s   i n   d a t a b a s e 
 
                         r e p o s i t o r y . s a v e _ e f f e c t i v e n e s s _ m e t r i c ( { 
 
                                 " t o o l _ i d " :   t o o l _ i d , 
 
                                 " m e t r i c _ t y p e " :   " w i n _ r a t e " , 
 
                                 " v a l u e " :   w i n _ r a t e , 
 
                                 " s t a r t _ d a t e " :   s t a r t _ d a t e , 
 
                                 " e n d _ d a t e " :   n o w , 
 
                                 " s i g n a l _ c o u n t " :   l e n ( s i g n a l s ) , 
 
                                 " o u t c o m e _ c o u n t " :   l e n ( o u t c o m e s ) 
 
                         } ) 
 
                         
 
                         r e p o s i t o r y . s a v e _ e f f e c t i v e n e s s _ m e t r i c ( { 
 
                                 " t o o l _ i d " :   t o o l _ i d , 
 
                                 " m e t r i c _ t y p e " :   " p r o f i t _ f a c t o r " , 
 
                                 " v a l u e " :   p r o f i t _ f a c t o r , 
 
                                 " s t a r t _ d a t e " :   s t a r t _ d a t e , 
 
                                 " e n d _ d a t e " :   n o w , 
 
                                 " s i g n a l _ c o u n t " :   l e n ( s i g n a l s ) , 
 
                                 " o u t c o m e _ c o u n t " :   l e n ( o u t c o m e s ) 
 
                         } ) 
 
                         
 
                         r e p o s i t o r y . s a v e _ e f f e c t i v e n e s s _ m e t r i c ( { 
 
                                 " t o o l _ i d " :   t o o l _ i d , 
 
                                 " m e t r i c _ t y p e " :   " e x p e c t e d _ p a y o f f " , 
 
                                 " v a l u e " :   e x p e c t e d _ p a y o f f , 
 
                                 " s t a r t _ d a t e " :   s t a r t _ d a t e , 
 
                                 " e n d _ d a t e " :   n o w , 
 
                                 " s i g n a l _ c o u n t " :   l e n ( s i g n a l s ) , 
 
                                 " o u t c o m e _ c o u n t " :   l e n ( o u t c o m e s ) 
 
                         } ) 
 
                 
 
                 r e t u r n   { " s t a t u s " :   " s u c c e s s " ,   " m e s s a g e " :   " O u t c o m e   r e g i s t e r e d   s u c c e s s f u l l y " } 
 
         e x c e p t   K e y E r r o r   a s   e : 
 
                 r a i s e   H T T P E x c e p t i o n ( s t a t u s _ c o d e = 4 0 4 ,   d e t a i l = f " S i g n a l   n o t   f o u n d :   { s t r ( e ) } " ) 
 
         e x c e p t   E x c e p t i o n   a s   e : 
 
                 r a i s e   H T T P E x c e p t i o n ( s t a t u s _ c o d e = 4 0 0 ,   d e t a i l = f " F a i l e d   t o   r e g i s t e r   o u t c o m e :   { s t r ( e ) } " ) 
 
 
 
 @ r o u t e r . g e t ( " / m e t r i c s / " ,   r e s p o n s e _ m o d e l = L i s t [ T o o l E f f e c t i v e n e s s ] ) 
 
 d e f   g e t _ e f f e c t i v e n e s s _ m e t r i c s ( 
 
         t o o l _ i d :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   s p e c i f i c   t o o l   I D " ) , 
 
         t i m e f r a m e :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   s p e c i f i c   t i m e f r a m e " ) , 
 
         i n s t r u m e n t :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   s p e c i f i c   i n s t r u m e n t " ) , 
 
         m a r k e t _ r e g i m e :   O p t i o n a l [ M a r k e t R e g i m e E n u m ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   m a r k e t   r e g i m e " ) , 
 
         f r o m _ d a t e :   O p t i o n a l [ d a t e t i m e . d a t e t i m e ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " S t a r t   d a t e   f o r   m e t r i c s   c a l c u l a t i o n " ) , 
 
         t o _ d a t e :   O p t i o n a l [ d a t e t i m e . d a t e t i m e ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " E n d   d a t e   f o r   m e t r i c s   c a l c u l a t i o n " ) , 
 
         d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) 
 
 ) : 
 
         " " " 
 
         R e t r i e v e   e f f e c t i v e n e s s   m e t r i c s   f o r   t r a d i n g   t o o l s   w i t h   o p t i o n a l   f i l t e r i n g 
 
         " " " 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         
 
         #   G e t   a l l   t o o l s   t h a t   m a t c h   f i l t e r s 
 
         i f   t o o l _ i d : 
 
                 t o o l s   =   [ r e p o s i t o r y . g e t _ t o o l ( t o o l _ i d ) ] 
 
                 i f   n o t   t o o l s [ 0 ] : 
 
                         r e t u r n   [ ] 
 
         e l s e : 
 
                 t o o l s   =   r e p o s i t o r y . g e t _ t o o l s ( l i m i t = 1 0 0 ) 
 
         
 
         r e s u l t   =   [ ] 
 
         f o r   t o o l   i n   t o o l s : 
 
                 #   G e t   m e t r i c s   f o r   t h i s   t o o l 
 
                 m e t r i c s _ d a t a   =   r e p o s i t o r y . g e t _ e f f e c t i v e n e s s _ m e t r i c s ( 
 
                         t o o l _ i d = t o o l . t o o l _ i d , 
 
                         t i m e f r a m e = t i m e f r a m e , 
 
                         i n s t r u m e n t = i n s t r u m e n t , 
 
                         m a r k e t _ r e g i m e = m a r k e t _ r e g i m e . v a l u e   i f   m a r k e t _ r e g i m e   e l s e   N o n e , 
 
                         f r o m _ d a t e = f r o m _ d a t e , 
 
                         t o _ d a t e = t o _ d a t e 
 
                 ) 
 
                 
 
                 i f   n o t   m e t r i c s _ d a t a : 
 
                         c o n t i n u e 
 
                 
 
                 #   G r o u p   m e t r i c s   b y   t y p e 
 
                 m e t r i c s _ d i c t   =   { } 
 
                 f o r   m e t r i c   i n   m e t r i c s _ d a t a : 
 
                         m e t r i c s _ d i c t [ m e t r i c . m e t r i c _ t y p e ]   =   m e t r i c 
 
                 
 
                 #   G e t   s i g n a l s   f o r   d a t e   r a n g e   c a l c u l a t i o n 
 
                 s i g n a l s   =   r e p o s i t o r y . g e t _ s i g n a l s ( 
 
                         t o o l _ i d = t o o l . t o o l _ i d , 
 
                         t i m e f r a m e = t i m e f r a m e , 
 
                         i n s t r u m e n t = i n s t r u m e n t , 
 
                         m a r k e t _ r e g i m e = m a r k e t _ r e g i m e . v a l u e   i f   m a r k e t _ r e g i m e   e l s e   N o n e , 
 
                         f r o m _ d a t e = f r o m _ d a t e , 
 
                         t o _ d a t e = t o _ d a t e , 
 
                         l i m i t = 1 0 0 0 
 
                 ) 
 
                 
 
                 i f   n o t   s i g n a l s : 
 
                         c o n t i n u e 
 
                 
 
                 #   C a l c u l a t e   o v e r a l l   s u c c e s s   r a t e 
 
                 s i g n a l _ i d s   =   [ s . s i g n a l _ i d   f o r   s   i n   s i g n a l s ] 
 
                 o u t c o m e s   =   [ ] 
 
                 f o r   s i g _ i d   i n   s i g n a l _ i d s : 
 
                         o u t c o m e s . e x t e n d ( r e p o s i t o r y . g e t _ o u t c o m e s _ f o r _ s i g n a l ( s i g _ i d ) ) 
 
                 
 
                 s u c c e s s _ c o u n t   =   s u m ( 1   f o r   o   i n   o u t c o m e s   i f   o . s u c c e s s ) 
 
                 s u c c e s s _ r a t e   =   s u c c e s s _ c o u n t   /   l e n ( o u t c o m e s )   i f   o u t c o m e s   e l s e   0 . 0 
 
                 
 
                 #   C a l c u l a t e   f i r s t   a n d   l a s t   s i g n a l   d a t e s 
 
                 t i m e s t a m p s   =   [ s . t i m e s t a m p   f o r   s   i n   s i g n a l s ] 
 
                 f i r s t _ s i g n a l _ d a t e   =   m i n ( t i m e s t a m p s )   i f   t i m e s t a m p s   e l s e   N o n e 
 
                 l a s t _ s i g n a l _ d a t e   =   m a x ( t i m e s t a m p s )   i f   t i m e s t a m p s   e l s e   N o n e 
 
                 
 
                 #   F o r m a t   m e t r i c s   f o r   A P I   r e s p o n s e 
 
                 f o r m a t t e d _ m e t r i c s   =   [ ] 
 
                 i f   ' w i n _ r a t e '   i n   m e t r i c s _ d i c t : 
 
                         f o r m a t t e d _ m e t r i c s . a p p e n d ( E f f e c t i v e n e s s M e t r i c ( 
 
                                 n a m e = " W i n   R a t e " , 
 
                                 v a l u e = m e t r i c s _ d i c t [ ' w i n _ r a t e ' ] . v a l u e , 
 
                                 d e s c r i p t i o n = " P e r c e n t a g e   o f   s u c c e s s f u l   s i g n a l s " 
 
                         ) ) 
 
                         
 
                 i f   ' p r o f i t _ f a c t o r '   i n   m e t r i c s _ d i c t : 
 
                         f o r m a t t e d _ m e t r i c s . a p p e n d ( E f f e c t i v e n e s s M e t r i c ( 
 
                                 n a m e = " P r o f i t   F a c t o r " , 
 
                                 v a l u e = m e t r i c s _ d i c t [ ' p r o f i t _ f a c t o r ' ] . v a l u e , 
 
                                 d e s c r i p t i o n = " R a t i o   o f   g r o s s   p r o f i t s   t o   g r o s s   l o s s e s " 
 
                         ) ) 
 
                         
 
                 i f   ' e x p e c t e d _ p a y o f f '   i n   m e t r i c s _ d i c t : 
 
                         f o r m a t t e d _ m e t r i c s . a p p e n d ( E f f e c t i v e n e s s M e t r i c ( 
 
                                 n a m e = " E x p e c t e d   P a y o f f " , 
 
                                 v a l u e = m e t r i c s _ d i c t [ ' e x p e c t e d _ p a y o f f ' ] . v a l u e , 
 
                                 d e s c r i p t i o n = " A v e r a g e   p r o f i t / l o s s   p e r   s i g n a l " 
 
                         ) ) 
 
                 
 
                 #   A d d   r e l i a b i l i t y   b y   r e g i m e   m e t r i c s   i f   a v a i l a b l e 
 
                 f o r   m e t r i c   i n   m e t r i c s _ d a t a : 
 
                         i f   m e t r i c . m e t r i c _ t y p e . s t a r t s w i t h ( ' r e l i a b i l i t y _ ' ) : 
 
                                 r e g i m e   =   m e t r i c . m e t r i c _ t y p e . s p l i t ( ' _ ' ) [ 1 ] 
 
                                 f o r m a t t e d _ m e t r i c s . a p p e n d ( E f f e c t i v e n e s s M e t r i c ( 
 
                                         n a m e = f " R e l i a b i l i t y   i n   { r e g i m e . c a p i t a l i z e ( ) }   M a r k e t " , 
 
                                         v a l u e = m e t r i c . v a l u e , 
 
                                         d e s c r i p t i o n = f " S u c c e s s   r a t e   i n   { r e g i m e }   m a r k e t   c o n d i t i o n s " 
 
                                 ) ) 
 
                 
 
                 t o o l _ e f f e c t i v e n e s s   =   T o o l E f f e c t i v e n e s s ( 
 
                         t o o l _ i d = t o o l . t o o l _ i d , 
 
                         m e t r i c s = f o r m a t t e d _ m e t r i c s , 
 
                         s i g n a l _ c o u n t = l e n ( s i g n a l s ) , 
 
                         f i r s t _ s i g n a l _ d a t e = f i r s t _ s i g n a l _ d a t e , 
 
                         l a s t _ s i g n a l _ d a t e = l a s t _ s i g n a l _ d a t e , 
 
                         s u c c e s s _ r a t e = s u c c e s s _ r a t e 
 
                 ) 
 
                 
 
                 r e s u l t . a p p e n d ( t o o l _ e f f e c t i v e n e s s ) 
 
         
 
         r e t u r n   r e s u l t 
 
 
 
 @ r o u t e r . d e l e t e ( " / t o o l / { t o o l _ i d } / d a t a / " ,   s t a t u s _ c o d e = 2 0 0 ) 
 
 d e f   c l e a r _ t o o l _ d a t a ( t o o l _ i d :   s t r ,   d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) ) : 
 
         " " " 
 
         C l e a r   a l l   d a t a   f o r   a   s p e c i f i c   t o o l   ( f o r   t e s t i n g   o r   r e s e t t i n g   p u r p o s e s ) 
 
         " " " 
 
         #   C l e a r   f r o m   i n - m e m o r y   t r a c k e r 
 
         o r i g i n a l _ s i g n a l _ c o u n t   =   l e n ( t r a c k e r . s i g n a l s ) 
 
         t r a c k e r . s i g n a l s   =   [ s   f o r   s   i n   t r a c k e r . s i g n a l s   i f   s . t o o l _ i d   ! =   t o o l _ i d ] 
 
         
 
         #   G e t   I D s   o f   r e m a i n i n g   s i g n a l s 
 
         r e m a i n i n g _ s i g n a l _ i d s   =   { s . i d   f o r   s   i n   t r a c k e r . s i g n a l s } 
 
         
 
         #   R e m o v e   o u t c o m e s   f o r   r e m o v e d   s i g n a l s 
 
         o r i g i n a l _ o u t c o m e _ c o u n t   =   l e n ( t r a c k e r . o u t c o m e s ) 
 
         t r a c k e r . o u t c o m e s   =   [ o   f o r   o   i n   t r a c k e r . o u t c o m e s   i f   o . s i g n a l _ i d   i n   r e m a i n i n g _ s i g n a l _ i d s ] 
 
         
 
         s i g n a l s _ r e m o v e d   =   o r i g i n a l _ s i g n a l _ c o u n t   -   l e n ( t r a c k e r . s i g n a l s ) 
 
         o u t c o m e s _ r e m o v e d   =   o r i g i n a l _ o u t c o m e _ c o u n t   -   l e n ( t r a c k e r . o u t c o m e s ) 
 
         
 
         #   C l e a r   f r o m   d a t a b a s e 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         d b _ s i g n a l s _ r e m o v e d ,   d b _ o u t c o m e s _ r e m o v e d   =   r e p o s i t o r y . d e l e t e _ t o o l _ d a t a ( t o o l _ i d ) 
 
         
 
         r e t u r n   { 
 
                 " s t a t u s " :   " s u c c e s s " , 
 
                 " s i g n a l s _ r e m o v e d " :   s i g n a l s _ r e m o v e d   +   d b _ s i g n a l s _ r e m o v e d , 
 
                 " o u t c o m e s _ r e m o v e d " :   o u t c o m e s _ r e m o v e d   +   d b _ o u t c o m e s _ r e m o v e d , 
 
                 " m e s s a g e " :   f " R e m o v e d   a l l   d a t a   f o r   t o o l   { t o o l _ i d } " 
 
         } 
 
 
 
 @ r o u t e r . g e t ( " / d a s h b o a r d - d a t a / " ,   r e s p o n s e _ m o d e l = D i c t ) 
 
 d e f   g e t _ d a s h b o a r d _ d a t a ( d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) ) : 
 
         " " " 
 
         G e t   a g g r e g a t e d   d a t a   s u i t a b l e   f o r   d a s h b o a r d   v i s u a l i z a t i o n 
 
         " " " 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         
 
         #   G e t   a l l   t o o l s 
 
         t o o l s   =   r e p o s i t o r y . g e t _ t o o l s ( l i m i t = 1 0 0 ) 
 
         t o o l _ i d s   =   [ t o o l . t o o l _ i d   f o r   t o o l   i n   t o o l s ] 
 
         
 
         #   G e t   s i g n a l s   f o r   a l l   t o o l s 
 
         a l l _ s i g n a l s   =   [ ] 
 
         f o r   t o o l _ i d   i n   t o o l _ i d s : 
 
                 a l l _ s i g n a l s . e x t e n d ( r e p o s i t o r y . g e t _ s i g n a l s ( t o o l _ i d = t o o l _ i d ,   l i m i t = 1 0 0 0 ) ) 
 
         
 
         i f   n o t   a l l _ s i g n a l s : 
 
                 r e t u r n   { 
 
                         " s u m m a r y " :   { 
 
                                 " t o t a l _ s i g n a l s " :   0 , 
 
                                 " t o t a l _ o u t c o m e s " :   0 , 
 
                                 " o v e r a l l _ s u c c e s s _ r a t e " :   0 
 
                         } , 
 
                         " f i l t e r s " :   { 
 
                                 " t o o l s " :   t o o l _ i d s , 
 
                                 " t i m e f r a m e s " :   [ ] , 
 
                                 " i n s t r u m e n t s " :   [ ] , 
 
                                 " r e g i m e s " :   [ ] 
 
                         } , 
 
                         " t o p _ p e r f o r m i n g _ t o o l s " :   [ ] 
 
                 } 
 
         
 
         #   E x t r a c t   u n i q u e   v a l u e s   f o r   f i l t e r s 
 
         t i m e f r a m e s   =   l i s t ( s e t ( s . t i m e f r a m e   f o r   s   i n   a l l _ s i g n a l s ) ) 
 
         i n s t r u m e n t s   =   l i s t ( s e t ( s . i n s t r u m e n t   f o r   s   i n   a l l _ s i g n a l s ) ) 
 
         r e g i m e s   =   l i s t ( s e t ( s . m a r k e t _ r e g i m e   f o r   s   i n   a l l _ s i g n a l s ) ) 
 
         
 
         #   C a l c u l a t e   t o o l   p e r f o r m a n c e 
 
         t o o l _ p e r f o r m a n c e   =   [ ] 
 
         f o r   t o o l   i n   t o o l s : 
 
                 #   G e t   w i n   r a t e   f r o m   s t o r e d   m e t r i c s 
 
                 l a t e s t _ m e t r i c s   =   r e p o s i t o r y . g e t _ l a t e s t _ t o o l _ m e t r i c s ( t o o l . t o o l _ i d ) 
 
                 w i n _ r a t e   =   l a t e s t _ m e t r i c s . g e t ( ' w i n _ r a t e ' ,   0 ) 
 
                 
 
                 #   G e t   s i g n a l   c o u n t 
 
                 s i g n a l s   =   r e p o s i t o r y . g e t _ s i g n a l s ( t o o l _ i d = t o o l . t o o l _ i d ,   l i m i t = 1 0 0 0 ) 
 
                 s i g n a l _ c o u n t   =   l e n ( s i g n a l s ) 
 
                 
 
                 #   G e t   o u t c o m e   c o u n t   a n d   s u c c e s s   r a t e 
 
                 o u t c o m e _ c o u n t   =   0 
 
                 s u c c e s s _ c o u n t   =   0 
 
                 f o r   s i g n a l   i n   s i g n a l s : 
 
                         o u t c o m e s   =   r e p o s i t o r y . g e t _ o u t c o m e s _ f o r _ s i g n a l ( s i g n a l . s i g n a l _ i d ) 
 
                         o u t c o m e _ c o u n t   + =   l e n ( o u t c o m e s ) 
 
                         s u c c e s s _ c o u n t   + =   s u m ( 1   f o r   o   i n   o u t c o m e s   i f   o . s u c c e s s ) 
 
                 
 
                 s u c c e s s _ r a t e   =   ( s u c c e s s _ c o u n t   /   o u t c o m e _ c o u n t )   *   1 0 0   i f   o u t c o m e _ c o u n t   >   0   e l s e   0 
 
                 
 
                 t o o l _ p e r f o r m a n c e . a p p e n d ( { 
 
                         " t o o l _ i d " :   t o o l . t o o l _ i d , 
 
                         " n a m e " :   t o o l . n a m e , 
 
                         " s i g n a l s _ c o u n t " :   s i g n a l _ c o u n t , 
 
                         " o u t c o m e s _ c o u n t " :   o u t c o m e _ c o u n t , 
 
                         " s u c c e s s _ r a t e " :   s u c c e s s _ r a t e , 
 
                         " w i n _ r a t e " :   w i n _ r a t e 
 
                 } ) 
 
         
 
         #   S o r t   b y   s u c c e s s   r a t e   f o r   t o p   p e r f o r m e r s 
 
         t o p _ t o o l s   =   s o r t e d ( t o o l _ p e r f o r m a n c e ,   k e y = l a m b d a   x :   x [ " s u c c e s s _ r a t e " ] ,   r e v e r s e = T r u e ) [ : 5 ] 
 
         
 
         #   C a l c u l a t e   o v e r a l l   m e t r i c s 
 
         t o t a l _ s i g n a l s   =   l e n ( a l l _ s i g n a l s ) 
 
         t o t a l _ o u t c o m e s   =   s u m ( t [ " o u t c o m e s _ c o u n t " ]   f o r   t   i n   t o o l _ p e r f o r m a n c e ) 
 
         o v e r a l l _ s u c c e s s _ r a t e   =   s u m ( t [ " s u c c e s s _ r a t e " ]   *   t [ " o u t c o m e s _ c o u n t " ]   f o r   t   i n   t o o l _ p e r f o r m a n c e )   /   t o t a l _ o u t c o m e s   i f   t o t a l _ o u t c o m e s   >   0   e l s e   0 
 
         
 
         r e t u r n   { 
 
                 " s u m m a r y " :   { 
 
                         " t o t a l _ s i g n a l s " :   t o t a l _ s i g n a l s , 
 
                         " t o t a l _ o u t c o m e s " :   t o t a l _ o u t c o m e s , 
 
                         " o v e r a l l _ s u c c e s s _ r a t e " :   o v e r a l l _ s u c c e s s _ r a t e 
 
                 } , 
 
                 " f i l t e r s " :   { 
 
                         " t o o l s " :   t o o l _ i d s , 
 
                         " t i m e f r a m e s " :   t i m e f r a m e s , 
 
                         " i n s t r u m e n t s " :   i n s t r u m e n t s , 
 
                         " r e g i m e s " :   r e g i m e s 
 
                 } , 
 
                 " t o p _ p e r f o r m i n g _ t o o l s " :   t o p _ t o o l s 
 
         } 
 
 
 
 @ r o u t e r . p o s t ( " / r e p o r t s / " ,   s t a t u s _ c o d e = 2 0 1 ,   r e s p o n s e _ m o d e l = D i c t ) 
 
 d e f   s a v e _ e f f e c t i v e n e s s _ r e p o r t ( 
 
         n a m e :   s t r   =   Q u e r y ( . . . ,   d e s c r i p t i o n = " N a m e   o f   t h e   r e p o r t " ) , 
 
         d e s c r i p t i o n :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " D e s c r i p t i o n   o f   t h e   r e p o r t " ) , 
 
         t o o l _ i d :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   s p e c i f i c   t o o l   I D " ) , 
 
         t i m e f r a m e :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   s p e c i f i c   t i m e f r a m e " ) , 
 
         i n s t r u m e n t :   O p t i o n a l [ s t r ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   s p e c i f i c   i n s t r u m e n t " ) , 
 
         m a r k e t _ r e g i m e :   O p t i o n a l [ M a r k e t R e g i m e E n u m ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " F i l t e r   b y   m a r k e t   r e g i m e " ) , 
 
         f r o m _ d a t e :   O p t i o n a l [ d a t e t i m e . d a t e t i m e ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " S t a r t   d a t e   f o r   m e t r i c s " ) , 
 
         t o _ d a t e :   O p t i o n a l [ d a t e t i m e . d a t e t i m e ]   =   Q u e r y ( N o n e ,   d e s c r i p t i o n = " E n d   d a t e   f o r   m e t r i c s " ) , 
 
         d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) 
 
 ) : 
 
         " " " 
 
         S a v e   a   n e w   e f f e c t i v e n e s s   r e p o r t 
 
         " " " 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         
 
         #   G e t   m e t r i c s   w i t h   f i l t e r s 
 
         m e t r i c s _ r e s u l t   =   g e t _ e f f e c t i v e n e s s _ m e t r i c s ( 
 
                 t o o l _ i d = t o o l _ i d , 
 
                 t i m e f r a m e = t i m e f r a m e , 
 
                 i n s t r u m e n t = i n s t r u m e n t , 
 
                 m a r k e t _ r e g i m e = m a r k e t _ r e g i m e , 
 
                 f r o m _ d a t e = f r o m _ d a t e , 
 
                 t o _ d a t e = t o _ d a t e , 
 
                 d b = d b 
 
         ) 
 
         
 
         #   C r e a t e   r e p o r t   d a t a 
 
         r e p o r t _ d a t a   =   { 
 
                 " m e t r i c s " :   [ m e t r i c . d i c t ( )   f o r   m e t r i c   i n   m e t r i c s _ r e s u l t ] , 
 
                 " g e n e r a t e d _ a t " :   d a t e t i m e . d a t e t i m e . u t c n o w ( ) . i s o f o r m a t ( ) , 
 
                 " s u m m a r y " :   { 
 
                         " t o o l _ c o u n t " :   l e n ( m e t r i c s _ r e s u l t ) , 
 
                         " t o t a l _ s i g n a l s " :   s u m ( m . s i g n a l _ c o u n t   f o r   m   i n   m e t r i c s _ r e s u l t ) , 
 
                         " a v g _ s u c c e s s _ r a t e " :   s u m ( m . s u c c e s s _ r a t e   f o r   m   i n   m e t r i c s _ r e s u l t )   /   l e n ( m e t r i c s _ r e s u l t )   i f   m e t r i c s _ r e s u l t   e l s e   0 
 
                 } 
 
         } 
 
         
 
         #   S a v e   f i l t e r s 
 
         f i l t e r s   =   { 
 
                 " t o o l _ i d " :   t o o l _ i d , 
 
                 " t i m e f r a m e " :   t i m e f r a m e , 
 
                 " i n s t r u m e n t " :   i n s t r u m e n t , 
 
                 " m a r k e t _ r e g i m e " :   m a r k e t _ r e g i m e . v a l u e   i f   m a r k e t _ r e g i m e   e l s e   N o n e , 
 
                 " f r o m _ d a t e " :   f r o m _ d a t e . i s o f o r m a t ( )   i f   f r o m _ d a t e   e l s e   N o n e , 
 
                 " t o _ d a t e " :   t o _ d a t e . i s o f o r m a t ( )   i f   t o _ d a t e   e l s e   N o n e 
 
         } 
 
         
 
         #   S a v e   r e p o r t 
 
         r e p o r t   =   r e p o s i t o r y . s a v e _ r e p o r t ( { 
 
                 " n a m e " :   n a m e , 
 
                 " d e s c r i p t i o n " :   d e s c r i p t i o n , 
 
                 " r e p o r t _ d a t a " :   r e p o r t _ d a t a , 
 
                 " f i l t e r s " :   f i l t e r s , 
 
                 " c r e a t e d _ a t " :   d a t e t i m e . d a t e t i m e . u t c n o w ( ) 
 
         } ) 
 
         
 
         r e t u r n   { 
 
                 " s t a t u s " :   " s u c c e s s " , 
 
                 " m e s s a g e " :   " R e p o r t   s a v e d   s u c c e s s f u l l y " , 
 
                 " r e p o r t _ i d " :   r e p o r t . i d 
 
         } 
 
 
 
 @ r o u t e r . g e t ( " / r e p o r t s / " ,   r e s p o n s e _ m o d e l = L i s t [ D i c t ] ) 
 
 d e f   g e t _ e f f e c t i v e n e s s _ r e p o r t s ( 
 
         s k i p :   i n t   =   Q u e r y ( 0 ,   d e s c r i p t i o n = " S k i p   i t e m s   f o r   p a g i n a t i o n " ) , 
 
         l i m i t :   i n t   =   Q u e r y ( 1 0 0 ,   d e s c r i p t i o n = " L i m i t   i t e m s   f o r   p a g i n a t i o n " ) , 
 
         d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) 
 
 ) : 
 
         " " " 
 
         G e t   a l l   s a v e d   e f f e c t i v e n e s s   r e p o r t s 
 
         " " " 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         r e p o r t s   =   r e p o s i t o r y . g e t _ r e p o r t s ( s k i p = s k i p ,   l i m i t = l i m i t ) 
 
         
 
         r e t u r n   [ 
 
                 { 
 
                         " i d " :   r e p o r t . i d , 
 
                         " n a m e " :   r e p o r t . n a m e , 
 
                         " d e s c r i p t i o n " :   r e p o r t . d e s c r i p t i o n , 
 
                         " f i l t e r s " :   r e p o r t . f i l t e r s , 
 
                         " c r e a t e d _ a t " :   r e p o r t . c r e a t e d _ a t 
 
                 } 
 
                 f o r   r e p o r t   i n   r e p o r t s 
 
         ] 
 
 
 
 @ r o u t e r . g e t ( " / r e p o r t s / { r e p o r t _ i d } " ,   r e s p o n s e _ m o d e l = D i c t ) 
 
 d e f   g e t _ e f f e c t i v e n e s s _ r e p o r t ( 
 
         r e p o r t _ i d :   i n t , 
 
         d b :   S e s s i o n   =   D e p e n d s ( g e t _ d b _ s e s s i o n ) 
 
 ) : 
 
         " " " 
 
         G e t   a   s p e c i f i c   e f f e c t i v e n e s s   r e p o r t   b y   I D 
 
         " " " 
 
         r e p o s i t o r y   =   T o o l E f f e c t i v e n e s s R e p o s i t o r y ( d b ) 
 
         r e p o r t   =   r e p o s i t o r y . g e t _ r e p o r t ( r e p o r t _ i d ) 
 
         
 
         i f   n o t   r e p o r t : 
 
                 r a i s e   H T T P E x c e p t i o n ( s t a t u s _ c o d e = 4 0 4 ,   d e t a i l = f " R e p o r t   n o t   f o u n d :   { r e p o r t _ i d } " ) 
 
         
 
         r e t u r n   { 
 
                 " i d " :   r e p o r t . i d , 
 
                 " n a m e " :   r e p o r t . n a m e , 
 
                 " d e s c r i p t i o n " :   r e p o r t . d e s c r i p t i o n , 
 
                 " f i l t e r s " :   r e p o r t . f i l t e r s , 
 
                 " c r e a t e d _ a t " :   r e p o r t . c r e a t e d _ a t , 
 
                 " r e p o r t _ d a t a " :   r e p o r t . r e p o r t _ d a t a 
 
         } 
 
 