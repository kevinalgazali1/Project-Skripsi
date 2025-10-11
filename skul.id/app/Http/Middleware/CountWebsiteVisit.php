<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Support\Facades\Session;
use App\Models\SiteVisit;

class CountWebsiteVisit
{
    public function handle($request, Closure $next)
    {
        if (!Session::has('site_visited')) {
            $visit = SiteVisit::first();
            if (!$visit) {
                $visit = SiteVisit::create(['count' => 1]);
            } else {
                $visit->increment('count');
            }

            Session::put('site_visited', true); // Menandai sesi ini sudah dihitung
        }

        return $next($request);
    }
}

