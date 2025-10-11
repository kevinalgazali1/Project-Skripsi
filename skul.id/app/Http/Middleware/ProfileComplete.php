<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class ProfileComplete
{
    /**
     * Handle an incoming request.
     *
     * @param  \Illuminate\Http\Request  $request
     * @param  \Closure  $next
     * @return mixed
     */
    public function handle(Request $request, Closure $next)
    {
        $user = Auth::user();

        if (!$user->profile_complete) {
            if ($user->role === 'alumnisiswa') {
                return redirect()->route('alumni-siswa.addProfile');
            } elseif ($user->role === 'mitra') {
                return redirect()->route('mitra.addProfile');
            }
        }

        return $next($request);
    }
}
