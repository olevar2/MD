"""
  init   module.

This module provides functionality for...
"""

" " " 
 
 A n a l y s i s   E n g i n e   S e r v i c e   A P I   v 1 
 
 
 
 T h i s   p a c k a g e   p r o v i d e s   A P I   e n d p o i n t s   f o r   t h e   e n h a n c e d   a n a l y s i s   e n g i n e , 
 
 i n c l u d i n g   N L P   a n a l y s i s ,   c o r r e l a t i o n   d e t e c t i o n ,   m a n i p u l a t i o n   d e t e c t i o n ,   m u l t i - a s s e t   s u p p o r t 
 
 c o m p o n e n t s   i m p l e m e n t e d   i n   P h a s e   7 ,   a n d   f e e d b a c k   l o o p   i n t e g r a t i o n   i m p l e m e n t e d   i n   P h a s e   8 . 
 
 " " " 
 
 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . a n a l y s i s _ r e s u l t s _ a p i   i m p o r t   r o u t e r   a s   a n a l y s i s _ r e s u l t s _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . m a r k e t _ r e g i m e _ a n a l y s i s   i m p o r t   r o u t e r   a s   m a r k e t _ r e g i m e _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . t o o l _ e f f e c t i v e n e s s _ a n a l y t i c s   i m p o r t   r o u t e r   a s   t o o l _ e f f e c t i v e n e s s _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . a d a p t i v e _ l a y e r   i m p o r t   r o u t e r   a s   a d a p t i v e _ l a y e r _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . s i g n a l _ q u a l i t y   i m p o r t   r o u t e r   a s   s i g n a l _ q u a l i t y _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . e n h a n c e d _ t o o l _ e f f e c t i v e n e s s   i m p o r t   r o u t e r   a s   e n h a n c e d _ t o o l _ r o u t e r 
 
 
 
 #   P h a s e   7   r o u t e r s 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . n l p _ a n a l y s i s   i m p o r t   r o u t e r   a s   n l p _ a n a l y s i s _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . c o r r e l a t i o n _ a n a l y s i s   i m p o r t   r o u t e r   a s   c o r r e l a t i o n _ a n a l y s i s _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . m a n i p u l a t i o n _ d e t e c t i o n   i m p o r t   r o u t e r   a s   m a n i p u l a t i o n _ d e t e c t i o n _ r o u t e r 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . m u l t i _ a s s e t   i m p o r t   r o u t e r   a s   m u l t i _ a s s e t _ r o u t e r 
 
 
 
 #   P h a s e   8   r o u t e r s 
 
 f r o m   a n a l y s i s _ e n g i n e . a p i . v 1 . f e e d b a c k   i m p o r t   r o u t e r   a s   f e e d b a c k _ r o u t e r 
 
 
 
 _ _ a l l _ _   =   [ 
 
         ' a n a l y s i s _ r e s u l t s _ r o u t e r ' , 
 
         ' m a r k e t _ r e g i m e _ r o u t e r ' , 
 
         ' t o o l _ e f f e c t i v e n e s s _ r o u t e r ' , 
 
         ' a d a p t i v e _ l a y e r _ r o u t e r ' , 
 
         ' s i g n a l _ q u a l i t y _ r o u t e r ' , 
 
         ' e n h a n c e d _ t o o l _ r o u t e r ' , 
 
         ' n l p _ a n a l y s i s _ r o u t e r ' , 
 
         ' c o r r e l a t i o n _ a n a l y s i s _ r o u t e r ' , 
 
         ' m a n i p u l a t i o n _ d e t e c t i o n _ r o u t e r ' , 
 
         ' m u l t i _ a s s e t _ r o u t e r ' , 
 
         ' f e e d b a c k _ r o u t e r ' 
 
 ] 
 
 