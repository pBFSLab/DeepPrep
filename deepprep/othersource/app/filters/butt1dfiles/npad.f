      function npad(n,margin)
      if(n.le.1)then
        npad=n
        return
      endif
      m=1
      do 2 j=1,12
      do 4 i=2,9
      npad=m*i
    4 if(i.ne.7.and.npad.ge.n+2*margin)return
    2 m=m*2
      npad=n
      return
      end

